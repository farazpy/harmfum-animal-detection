from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
from PIL import Image
import time
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change to a secure random string

# MySQL configuration
db_config = {
    'host': 'localhost',
    'user': 'webmaster',
    'password': 'iFFP@1692',
    'database': 'farm_surveillance'
}

# Function to connect to MySQL
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as e:
        print(f"[ERROR] Failed to connect to MySQL: {e}")
        return None

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            conn.close()
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            conn.close()
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed', 'danger')
        return redirect(url_for('login'))
    
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = %s", (session['user_id'],))
    username = cursor.fetchone()[0]
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}
    cursor.execute("SELECT * FROM detection_logs ORDER BY detected_time DESC LIMIT 10")
    detection_logs = cursor.fetchall()
    log_columns = [desc[0] for desc in cursor.description]
    logs = [dict(zip(log_columns, row)) for row in detection_logs]
    cursor.close()
    conn.close()
    
    return render_template('dashboard.html', username=username, config=config, logs=logs)

@app.route('/dashboard/<page>')
def dashboardPages(page):
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed', 'danger')
        return redirect(url_for('login'))
    
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = %s", (session['user_id'],))
    username = cursor.fetchone()[0]
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}
    cursor.execute("SELECT * FROM detection_logs ORDER BY detected_time DESC LIMIT 10")
    detection_logs = cursor.fetchall()
    log_columns = [desc[0] for desc in cursor.description]
    logs = [dict(zip(log_columns, row)) for row in detection_logs]
    
    images = None    
    if page == "blacklist-objects":
        harmful_dir = config['harmful_dir']
        if not os.path.exists(harmful_dir):
            os.makedirs(harmful_dir)
            images = []
        else:
            images = [f for f in os.listdir(harmful_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for log in logs:
        if log.get('footage_path'):
            log['footage_path'] = log['footage_path'].replace('flask/', '')
    
    cursor.close()
    conn.close()
    
    return render_template(f'{page}.html', username=username, config=config, logs=logs, images=images)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))




    


@app.route('/delete_log/<int:log_id>', methods=['POST'])
def delete_log(log_id):
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    connection = get_db_connection()
    if not connection:
        flash('Database connection failed', 'danger')
        return redirect(url_for('dashboardPages', page='detection-logs'))
    
    try:
        cursor = connection.cursor()
        # Retrieve the footage_path before deleting the log
        cursor.execute("SELECT footage_path FROM detection_logs WHERE id = %s", (log_id,))
        result = cursor.fetchone()
        
        # Delete the log from the database
        cursor.execute("DELETE FROM detection_logs WHERE id = %s", (log_id,))
        connection.commit()
        
        # Delete the associated footage file if it exists
        if result and result[0]:  # Check if footage_path exists
            footage_path = result[0]
            if os.path.exists(footage_path):
                os.remove(footage_path)
                print(f"[INFO] Deleted footage file: {footage_path}")
            else:
                print(f"[WARNING] Footage file not found: {footage_path}")
        
        flash('Log and associated footage deleted successfully!', 'success')
    except mysql.connector.Error as e:
        flash(f'Error deleting log: {e}', 'danger')
    except OSError as e:
        flash(f'Log deleted, but failed to delete footage file: {e}', 'warning')
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('dashboardPages', page='detection-logs'))

# Handling forms below

@app.route('/rename_blacklist_image/<filename>', methods=['POST'])
def rename_blacklist_image(filename):
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}
    cursor.close()
    conn.close()

    old_file_path = os.path.join(config['harmful_dir'], filename)
    if not os.path.exists(old_file_path):
        flash('Image not found', 'danger')
        return redirect(url_for('dashboardPages', page='blacklist-objects'))

    new_filename = request.form.get('new_filename')
    if not new_filename:
        flash('No new filename provided', 'danger')
        return redirect(url_for('dashboardPages', page='blacklist-objects'))

    _, original_extension = os.path.splitext(filename)
    if not new_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        new_filename += original_extension

    new_filename = secure_filename(new_filename)
    new_file_path = os.path.join(config['harmful_dir'], new_filename)

    if os.path.exists(new_file_path):
        flash('An image with this name already exists', 'danger')
        return redirect(url_for('dashboardPages', page='blacklist-objects'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE known_faces SET filename = %s WHERE filename = %s", (new_filename, filename))
        conn.commit()
        cursor.close()
        conn.close()
        os.rename(old_file_path, new_file_path)
        flash(f'Image renamed to {new_filename} successfully', 'success')
    except (OSError, mysql.connector.Error) as e:
        flash(f'Failed to rename image: {e}', 'danger')

    return redirect(url_for('dashboardPages', page='blacklist-objects'))

@app.route('/add_blacklist_image', methods=['POST'])
def add_blacklist_image():
    # This route is now redundant; use upload_blacklist_item instead
    flash('Use the unified upload form for all items.', 'warning')
    return redirect(url_for('dashboardPages', page='blacklist-objects'))

@app.route('/delete_blacklist_image/<filename>', methods=['POST'])
def delete_blacklist_image(filename):
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}
    cursor.close()
    conn.close()

    file_path = os.path.join(config['harmful_dir'], filename)
    if os.path.exists(file_path):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM known_faces WHERE filename = %s", (filename,))
        conn.commit()
        cursor.close()
        conn.close()
        os.remove(file_path)
        flash(f'Image {filename} deleted successfully', 'success')
    else:
        flash('Image not found', 'danger')
    
    return redirect(url_for('dashboardPages', page='blacklist-objects'))

# New Unified Upload Route
@app.route('/upload_blacklist_item', methods=['POST'])
def upload_blacklist_item():
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}
    cursor.close()
    conn.close()

    if 'image' not in request.files or 'item_name' not in request.form:
        flash('Missing file or item name', 'danger')
        return redirect(url_for('dashboardPages', page='blacklist-objects'))
    
    file = request.files['image']
    item_name = request.form['item_name']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboardPages', page='blacklist-objects'))
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Generate a unique filename with the item name
        filename = secure_filename(f"{item_name}_{int(time.time())}.{file.filename.rsplit('.', 1)[1].lower()}")
        file_path = os.path.join(config['harmful_dir'], filename)
        
        # Save the file temporarily
        file.save(file_path)
        
        # Detect if it's a face or object
        img = cv2.imread(file_path)
        if img is None:
            flash('Failed to load image for detection', 'danger')
            os.remove(file_path)
            return redirect(url_for('dashboardPages', page='blacklist-objects'))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        is_face = len(faces) > 0
        
        # Crop to square (as in your existing logic)
        pil_img = Image.open(file_path)
        width, height = pil_img.size
        square_size = min(width, height)
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        cropped_img = pil_img.crop((left, top, right, bottom))
        cropped_img.save(file_path, quality=95)

        # Register in known_faces table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO known_faces (filename) VALUES (%s)", (filename,))
        conn.commit()
        cursor.close()
        conn.close()

        if is_face:
            flash(f'Face image for {item_name} uploaded and detected successfully', 'success')
        else:
            flash(f'Object image for {item_name} uploaded successfully (no face detected)', 'warning')
    else:
        flash('Invalid file type. Only PNG, JPG, and JPEG allowed.', 'danger')
    
    return redirect(url_for('dashboardPages', page='blacklist-objects'))

@app.route('/edit_config', methods=['GET', 'POST'])
def edit_config():
    if 'user_id' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))

    conn = get_db_connection()
    if not conn:
        flash('Database connection failed', 'danger')
        return redirect(url_for('dashboard'))
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config_data = cursor.fetchone()
    config_columns = [desc[0] for desc in cursor.description]
    config = dict(zip(config_columns, config_data)) if config_data else {}

    if request.method == 'POST':
        proto_path = request.form['proto_path']
        model_path = request.form['model_path']
        siren_path = request.form['siren_path']
        harmful_dir = request.form['harmful_dir']
        ip_webcam_url = request.form['ip_webcam_url']
        frame_width = int(request.form['frame_width'])
        frame_height = int(request.form['frame_height'])
        conf_thresh = float(request.form['conf_thresh'])
        template_threshold = float(request.form['template_threshold'])
        frame_skip = int(request.form['frame_skip'])
        alert_cooldown = float(request.form['alert_cooldown'])
        farm_owner_number = request.form['farm_owner_number']
        twilio_sid = request.form['twilio_sid']
        twilio_token = request.form['twilio_token']
        twilio_number = request.form['twilio_number']
        req_classes = request.form['req_classes']
        camera_name = request.form['camera_name']

        try:
            cursor.execute("""
                UPDATE config SET 
                    proto_path = %s, model_path = %s, siren_path = %s, harmful_dir = %s,
                    ip_webcam_url = %s, frame_width = %s, frame_height = %s, conf_thresh = %s,
                    template_threshold = %s, frame_skip = %s, alert_cooldown = %s,
                    farm_owner_number = %s, twilio_sid = %s, twilio_token = %s, twilio_number = %s,
                    req_classes = %s, camera_name = %s
                WHERE id = 1
            """, (
                proto_path, model_path, siren_path, harmful_dir, ip_webcam_url,
                frame_width, frame_height, conf_thresh, template_threshold, frame_skip,
                alert_cooldown, farm_owner_number, twilio_sid, twilio_token, twilio_number,
                req_classes, camera_name
            ))
            conn.commit()
            flash('Configuration updated successfully', 'success')
            return redirect(url_for('dashboardPages', page='configuration'))
        except mysql.connector.Error as e:
            conn.rollback()
            flash(f'Failed to update configuration: {e}', 'danger')
        except ValueError as e:
            flash(f'Invalid input: {e}', 'danger')

    cursor.close()
    conn.close()
    return render_template('configuration.html', config=config)

if __name__ == '__main__':
    app.run(debug=True)