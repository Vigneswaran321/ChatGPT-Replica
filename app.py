from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
import PyPDF2
import cohere
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

co = cohere.Client("COHERE_API_KEY")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  
        return text

def process_document(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    return chunks

def get_answer(query, chunks):
    vectorizer = TfidfVectorizer().fit_transform([query] + chunks)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity([vectors[0]], vectors[1:])
    top_indices = np.argsort(cosine_matrix[0])[::-1][:5]
    top_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(top_chunks)

    response = co.chat(
        message=f"""
        System Prompt:
        You are ChatGPT like Assistant, A file-based assistant that answers user questions based on uploaded documents.
        User Query: {query}
        Relevant Context:
        {context}
        """,
        model="command-r-plus"
    )
    return response.text

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        user = User(email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    uploaded_filename = session.get('uploaded_filename', '')
    answer = None
    error = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename == '':
                error = "No selected file"
            elif not file.filename.endswith('.pdf'):
                error = "Only PDF files are allowed"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath)
                chunks = process_document(text)
                session['document_chunks'] = json.dumps(chunks)
                session['uploaded_filename'] = filename
                uploaded_filename = filename

        if 'query' in request.form:
            query = request.form['query']
            if 'document_chunks' not in session:
                error = "Please upload a file first."
            else:
                chunks = json.loads(session['document_chunks'])
                answer = get_answer(query, chunks)

    return render_template('dashboard.html', answer=answer, uploaded_filename=uploaded_filename, error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('document_chunks', None)
    session.pop('uploaded_filename', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)