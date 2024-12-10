from flask import Flask, render_template, request, flash, redirect, session, url_for, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from datetime import datetime, date, timedelta
from flask_session import Session
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, HiddenField, SelectField, PasswordField, DateField
from wtforms.validators import DataRequired, Length, Optional, EqualTo, ValidationError, Email
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload
import os
import csv
import io
import numpy as np
import joblib

app = Flask(__name__, static_folder='static')

app.config['SECRET_KEY'] = '4046bde895cc19ca9cbd373a'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


db = SQLAlchemy(app)
migrate = Migrate(app, db)
Session(app)


# Load the logistic regression model
model = joblib.load('multioutput_rf.pkl')
barangays = [
            'Aganan', 'Amparo', 'Anilao', 'Balabag', 'Cabugao Norte', 'Cabugao Sur',
            'Jibao-an', 'Mali-ao', 'Pagsanga-an', 'Pal-agon', 'Pandac', 'Purok 1',
            'Purok 2', 'Purok 3', 'Purok 4', 'Tigum', 'Ungka 1', 'Ungka 2'
        ]

# Association Table for BHW and Barangays (many-to-many relationship)
bhw_barangay = db.Table('bhw_barangay',
    db.Column('bhw_id', db.Integer, db.ForeignKey('admin.id'), primary_key=True),
    db.Column('barangay_id', db.Integer, db.ForeignKey('barangay.id'), primary_key=True)
)

class Barangay(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)

    def __repr__(self):
        return f'Barangay {self.name}'
    
    
# User Class
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey('barangay.id'), nullable=True)

    # Relationship with Barangay
    barangay = db.relationship('Barangay', backref='users')

    # One-to-many relationship with Household
    households = db.relationship(
        "Household",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # Relationship for PredictionData
    predictions = db.relationship(
        "PredictionData",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f'<User {self.username}>'

# Child Class
class Child(db.Model):
    __tablename__ = 'child'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey('barangay.id'), nullable=False) 
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    parent_id = db.Column(db.Integer, nullable=False)

    # Relationships
    predictions = db.relationship("PredictionData", back_populates="child")
    user = db.relationship("User", backref="children")
    barangay = db.relationship("Barangay", backref="children")  

    def __repr__(self):
        return f'<Child {self.first_name} {self.last_name}>'


class ChildForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Household Class
class Household(db.Model):
    __tablename__ = 'household'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    address = db.Column(db.String(100), nullable=False)
    mother_first_name = db.Column(db.String(100), nullable=False)
    mother_last_name = db.Column(db.String(100), nullable=False)
    father_first_name = db.Column(db.String(100), nullable=False)
    father_last_name = db.Column(db.String(100), nullable=False)
    mother_age = db.Column(db.Integer, nullable=False)
    father_age = db.Column(db.Integer, nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey('barangay.id'))

    # Relationship with User
    user = db.relationship("User", back_populates="households")

    # Optional relationship with Barangay if needed
    barangay = db.relationship("Barangay", backref="households", foreign_keys=[barangay_id])

    # Relationship with PredictionData
    predictions = db.relationship(
        "PredictionData",
        back_populates="household",
        cascade="all, delete"
    )
    def __repr__(self):
        return f'<Household {self.id}>'

class HouseholdForm(FlaskForm):
    mother_first_name = StringField('Mother\'s First Name', validators=[DataRequired()])
    mother_last_name = StringField('Mother\'s Last Name', validators=[DataRequired()])
    father_first_name = StringField('Father\'s First Name', validators=[DataRequired()])
    father_last_name = StringField('Father\'s Last Name', validators=[DataRequired()])
    mother_birthdate = DateField('Mother\'s Birthdate', format='%Y-%m-%d', validators=[DataRequired()])
    father_birthdate = DateField('Father\'s Birthdate', format='%Y-%m-%d', validators=[DataRequired()])
    address = StringField('Address', validators=[DataRequired()])
    submit = SubmitField('Save Household')

# PredictionData Class
class PredictionData(db.Model):
    __tablename__ = 'prediction_data'

    id = db.Column(db.Integer, primary_key=True)
    child_id = db.Column(db.Integer, db.ForeignKey('child.id', ondelete='CASCADE'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    admin_id = db.Column(db.Integer, nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey('barangay.id'))
    household_id = db.Column(db.Integer, db.ForeignKey('household.id', ondelete='CASCADE'), nullable=True)

    child_first_name = db.Column(db.String(50), nullable=False)
    child_last_name = db.Column(db.String(50), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Prediction outcome fields for malnutrition metrics
    weight_status = db.Column(db.String(50), nullable=False)         
    height_status = db.Column(db.String(50), nullable=False)        
    weight_length_status = db.Column(db.String(50), nullable=False)  

    # Data fields specific to prediction
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(1), nullable=False)
    vitamin_a = db.Column(db.String(3), nullable=False)
    birth_order = db.Column(db.Integer, nullable=False)
    breastfeeding = db.Column(db.String(3), nullable=False)
    comorbidity_status = db.Column(db.String(3), nullable=False)
    type_of_birth = db.Column(db.String(50), nullable=False)
    size_at_birth = db.Column(db.String(50), nullable=False)
    dietary_diversity_score = db.Column(db.String(50), nullable=False)
    mothers_age = db.Column(db.Integer, nullable=False)
    mothers_education_level = db.Column(db.String(50), nullable=False)
    fathers_education_level = db.Column(db.String(50), nullable=False)
    womens_autonomy_tertiles = db.Column(db.String(50), nullable=False)
    toilet_facility = db.Column(db.String(20), nullable=False)
    source_of_drinking_water = db.Column(db.String(50), nullable=False)
    bmi_of_mother = db.Column(db.String(50), nullable=False)
    number_of_children_under_five = db.Column(db.Integer, nullable=False)
    household_size = db.Column(db.Integer, nullable=False)
    mothers_working_status = db.Column(db.String(50), nullable=False)
    wealth_quantile = db.Column(db.String(20), nullable=False) 
    prediction_result = db.Column(db.String(255), nullable=False)

    user = db.relationship("User", back_populates="predictions")
    household = db.relationship("Household", back_populates="predictions")
    child = db.relationship("Child", back_populates="predictions")

    def __repr__(self):
        return f'<PredictionData {self.id} - {self.child_first_name} {self.child_last_name} - {self.prediction_date}>'




class PredictionForm(FlaskForm):
    child_first_name = StringField("Child's First Name", validators=[DataRequired()])
    child_last_name = StringField("Child's Last Name", validators=[DataRequired()])
    age = DateField("Child's Birthdate", format='%Y-%m-%d', validators=[DataRequired()])
    sex = SelectField('Sex', choices=[('M', 'Male'), ('F', 'Female')], validators=[DataRequired()])
    vitamin_a = SelectField('Vitamin A', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    birth_order = IntegerField('Birth Order', validators=[DataRequired()])
    breastfeeding = SelectField('Breastfeeding', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])  # Only Yes/No choices
    comorbidity_status = SelectField('Comorbidity Status', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    type_of_birth = SelectField('Type of Birth', choices=[('Singleton', 'Singleton'), ('Multiple', 'Multiple')], validators=[DataRequired()])
    size_at_birth = SelectField('Size at Birth', choices=[
        ('Smaller than average', 'Smaller than average'),
        ('Average', 'Average'),
        ('Larger than average', 'Larger than average')
    ], validators=[DataRequired()])
    dietary_diversity_score = SelectField('Dietary Diversity Score', choices=[
        ('Below Minimum', 'Below Minimum'),
        ('Minimum', 'Minimum Requirement'),
        ('Above Minimum', 'Maximum Requirement')
    ], validators=[DataRequired()])
    mothers_age = IntegerField("Mother's Age", validators=[DataRequired()])
    mothers_education_level = SelectField("Mother's Education Level", choices=[
        ('No Proper Education', 'No Proper Education'),
        ('Elementary', 'Elementary'), ('Highschool', 'Highschool'), ('College', 'College')
    ], validators=[DataRequired()])
    fathers_education_level = SelectField("Father's Education Level", choices=[
        ('No Proper Education', 'No Proper Education'),
        ('Elementary', 'Elementary'), ('Highschool', 'Highschool'), ('College', 'College')
    ], validators=[DataRequired()])
    womens_autonomy_tertiles = SelectField("Women's Autonomy Tertiles", choices=[
        ('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')
    ], validators=[DataRequired()])
    toilet_facility = SelectField("Toilet Facility", choices=[
        ('Unimproved', 'Unimproved'), ('Improved', 'Improved')
    ], validators=[DataRequired()])
    source_of_drinking_water = SelectField("Source of Drinking Water", choices=[
        ('Clean', 'Clean'), ('Contaminated', 'Contaminated'), ('Untested', 'Untested')
    ], validators=[DataRequired()])
    bmi_of_mother = SelectField("BMI of Mother", choices=[
        ('Underweight', 'Underweight'), ('Normal', 'Normal'), ('Overweight', 'Overweight'), ('Obese', 'Obese')
    ], validators=[DataRequired()])
    number_of_children_under_five = IntegerField("Number of Children Under Five", validators=[DataRequired()])
    mothers_working_status = SelectField("Mother's Working Status", choices=[
        ('Working', 'Working'), ('Not Working', 'Not Working')
    ], validators=[DataRequired()])
    household_size = IntegerField("Household Size", validators=[DataRequired()])
    wealth_quantile = SelectField("Wealth Quantile", choices=[
        ('Poor', 'Poor'), ('Low Income', 'Low Income'), ('Low Middle Income', 'Low Middle Income'),
        ('Middle Income', 'Middle Income'), ('Upper Middle Income', 'Upper Middle Income'),
        ('Upper Income', 'Upper Income'), ('Rich', 'Rich')
    ], validators=[DataRequired()])

    residence = SelectField('Residence', choices=[('Rural', 'Rural'), ('Urban', 'Urban')], validators=[DataRequired()])
    submit = SubmitField('Submit')



class NewUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=50)])
    barangay = StringField('Barangay', render_kw={'readonly': True})  
    submit = SubmitField('Add User')
  
    
# BHW Admin Class    
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    barangay_id = db.Column(db.Integer, db.ForeignKey('barangay.id'), nullable=False)
    barangay = db.relationship('Barangay', backref='admins')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# BHW Admin Form with dynamic barangay choices
class CreateAdminForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    barangay = SelectField('Primary Barangay', validators=[DataRequired()], choices=[
        ('Aganan', 'Aganan'), ('Amparo', 'Amparo'), ('Anilao', 'Anilao'), ('Balabag', 'Balabag'),
        ('Cabugao Norte', 'Cabugao Norte'), ('Cabugao Sur', 'Cabugao Sur'), ('Jibao-an', 'Jibao-an'),
        ('Mali-ao', 'Mali-ao'), ('Pagsanga-an', 'Pagsanga-an'), ('Pal-agon', 'Pal-agon'),
        ('Pandac', 'Pandac'), ('Purok 1', 'Purok 1'), ('Purok 2', 'Purok 2'),
        ('Purok 3', 'Purok 3'), ('Purok 4', 'Purok 4'), ('Tigum', 'Tigum'),
        ('Ungka 1', 'Ungka 1'), ('Ungka 2', 'Ungka 2')
    ])
    submit = SubmitField('Create BHW')

# Update BHW 
class UpdateBHWForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password (Leave blank to keep current password)', validators=[Optional()])
    barangay = SelectField('Barangay', choices=[], validators=[DataRequired()])
    submit = SubmitField('Update BHW')

class RCHU(db.Model):
    __tablename__ = 'rchu'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    def __repr__(self):
        return f'<RCHU {self.username}>'

# Flaskform for Creating RHU
class CreateRCHUForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=4, max=25)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6)
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Create RCHU')

    def validate_username(self, username):
        existing_rchu = RCHU.query.filter_by(username=username.data).first()
        if existing_rchu:
            raise ValidationError('This username is already taken. Please choose a different one.')

class UpdateRCHUForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('New Password', validators=[Length(min=6, message="Optional")])
    confirm_password = PasswordField('Confirm Password', validators=[EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Update RCHU')

    def validate_username(self, username):
        existing_user = RCHU.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError('This username is already taken.')


# Association Table
bhw_barangay = db.Table('bhw_barangay',
    db.Column('bhw_id', db.Integer, db.ForeignKey('admin.id'), primary_key=True),
    db.Column('barangay_id', db.Integer, db.ForeignKey('barangay.id'), primary_key=True), extend_existing=True
)

class DummyForm(FlaskForm):
    hidden_tag = HiddenField('hidden_tag')

def has_access(self, barangay_name):
    return any(barangay.name == barangay_name for barangay in self.barangays)

@app.before_request
def insert_barangays():
    barangays = [
        'Aganan', 'Amparo', 'Anilao', 'Balabag', 'Cabugao Norte', 'Cabugao Sur',
        'Jibao-an', 'Mali-ao', 'Pagsanga-an', 'Pal-agon', 'Pandac', 'Purok 1',
        'Purok 2', 'Purok 3', 'Purok 4', 'Tigum', 'Ungka 1', 'Ungka 2'
    ]

    # Check if there are existing barangays before inserting
    existing_barangays = Barangay.query.first()
    if not existing_barangays:
        for name in barangays:
            new_barangay = Barangay(name=name)
            db.session.add(new_barangay)
        db.session.commit()
        
# Initialize the database
def create_db_tables():
    with app.app_context():
        db.create_all()
        print("Database tables created.")

create_db_tables()


# Custom Jinja2 filter to format numbers
def format_number(value):
    if value.is_integer():
        return int(value)
    return round(value, 1)

app.jinja_env.filters['format_number'] = format_number

# Main index page
@app.route('/')
def index():
    return render_template('System4/index.html')

# User Type
@app.route('/user-type-redirect', methods=['POST'])
def userTypeRedirect():
    user_type = request.form.get('user-type')
    
    if user_type == 'guardian':
        return redirect(url_for('userIndex'))
    elif user_type == 'health_worker':
        return redirect(url_for('adminIndex'))
    elif user_type == 'rch':
        return redirect(url_for('rchuLogin'))
    else:
        flash('Invalid user type selected.', 'danger')
        return redirect(url_for('index'))


#-------------------------Admin Area---------------------------------------

def authenticate_admin(username, password):
    admin = Admin.query.filter_by(username=username).first() 
    if admin and check_password_hash(admin.password_hash, password):
        return admin.id
    return None

# Admin login
@app.route('/admin/', methods=['GET', 'POST'])
def adminIndex():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Authenticate the user (Admin/BHW)
        admin_id = authenticate_admin(username, password)
        
        if admin_id:
            session['admin_id'] = admin_id
            session['role'] = 'BHW'
            return redirect(url_for('adminDashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('admin/login.html')

# Route to view all barangays
@app.route('/admin/barangays')
def adminBarangays():
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))
    
    admin_id = session.get('admin_id')
    admin = Admin.query.get(admin_id)
    
    if not admin:
        flash('Admin account not found.', 'danger')
        return redirect(url_for('adminIndex'))
    
    barangays = admin.barangays  # Passing full Barangay objects

    return render_template('admin/barangays.html', barangays=barangays)


# Admin view user profiles filtered by barangay
@app.route('/admin/user_profiles')
def adminUserProfiles():
    # Check if admin is logged in
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))
    
    # Retrieve the admin and their assigned barangay
    admin_id = session['admin_id']
    admin = Admin.query.get(admin_id)
    if not admin or not admin.barangay:
        flash("No barangay assigned to this admin.", "danger")
        return redirect(url_for('adminDashboard'))
    
    # Get the barangay instance assigned to the admin
    barangay_instance = admin.barangay

    # Fetch users in the admin's assigned barangay
    users = User.query.filter(User.barangay_id == barangay_instance.id).all()
    
    if not users:
        flash(f'No users found in {barangay_instance.name}.', 'info')
    
    # Render user_profiles template with users in the admin's barangay
    return render_template('admin/user_profiles.html', users=users, barangay=barangay_instance.name)


# Admin View Prediction Data
@app.route('/admin/predictions')
def adminPredictionData():
    if 'admin_id' in session:
        predictions = PredictionData.query.all()
        users = {user.id: user for user in User.query.all()}
        
        prediction_list = []
        for prediction in predictions:
            user = users.get(prediction.parent_id)
            prediction_list.append({
                'id': prediction.id,
                'child_first_name': prediction.child_first_name,
                'child_last_name': prediction.child_last_name,
                'age': prediction.age,
                'sex': prediction.sex,
                'prediction_result': prediction.prediction_result,
                'household_id': prediction.household_id 
            })
        
        return render_template('admin/viewpredictions.html', predictions=prediction_list)
    else:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

# View predictions filtered by barangay
@app.route('/admin/predictions/<string:barangay>', methods=['GET'])
def adminBarangayPredictions(barangay):
    if 'admin_id' in session:
        households = Household.query.filter_by(address=barangay).all()
        predictions = []
        for household in households:
            household_predictions = PredictionData.query.filter_by(household_id=household.id).all()
            for prediction in household_predictions:
                predictions.append({
                    'id': prediction.id,
                    'child_first_name': prediction.child_first_name,
                    'child_last_name': prediction.child_last_name,
                    'age': prediction.age,
                    'sex': prediction.sex,
                    'prediction_result': prediction.prediction_result,
                    'household_address': household.address,
                    'mother_first_name': household.mother_first_name,
                    'mother_last_name': household.mother_last_name
                })
        return render_template('admin/barangay_predictions.html', predictions=predictions, barangay=barangay)
    else:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))


# Admin add new user
@app.route('/admin/add_user', methods=['GET', 'POST'])
def adminAddUser():
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    form = NewUserForm()

    # Fetch the current admin’s barangay
    admin_id = session['admin_id']
    admin = Admin.query.get(admin_id)
    if not admin or not admin.barangay:
        flash("No barangay assigned to this admin.", "danger")
        return redirect(url_for('adminDashboard'))

    # Pre-fill the barangay field with the admin's barangay
    form.barangay.data = admin.barangay.name

    if form.validate_on_submit():
        new_user = User(
            username=form.username.data,
            barangay_id=admin.barangay.id  
        )
        new_user.set_password(form.password.data)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('New user created successfully!', 'success')
            
            # Redirect to household information form after creating user
            return redirect(url_for('adminAddHousehold', user_id=new_user.id))
        
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'danger')

    return render_template('admin/admin_add_user.html', form=form)


# Admin edit user
@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
def adminEditUser(user_id):
    if 'admin_id' in session:
        user = User.query.get_or_404(user_id)
        form = NewUserForm(obj=user)

        if form.validate_on_submit():
            user.username = form.username.data
            user.barangay = form.barangay.data
            if form.password.data:
                user.set_password(form.password.data)
            try:
                db.session.commit()
                flash('User profile updated successfully!', 'success')
                return redirect(url_for('adminUserProfiles', barangay=user.barangay))
            except IntegrityError:
                db.session.rollback()
                flash('Username already exists. Please choose a different username.', 'danger')

        return render_template('admin/admin_edit_user.html', form=form, user=user)
    else:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

# Admin Delete User
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def adminDeleteUser(user_id):
    # Check for admin access
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))
    
    # Fetch the user by ID
    user = User.query.get(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('adminUserProfiles'))

    try:
        # Delete all related PredictionData records
        predictions = PredictionData.query.filter(PredictionData.parent_id == user_id).all()
        for prediction in predictions:
            db.session.delete(prediction)
        
        # Delete all related children records
        children = Child.query.filter(Child.user_id == user_id).all()
        for child in children:
            db.session.delete(child)
        
        # Delete all related households
        households = Household.query.filter(Household.user_id == user_id).all()
        for household in households:
            db.session.delete(household)
        
        # Finally, delete the user
        db.session.delete(user)
        db.session.commit()  # Commit deletions

        flash('User and all related data deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()  # Rollback in case of error
        flash(f'Error deleting user: {e}', 'danger')

    return redirect(url_for('adminUserProfiles'))

@app.route('/admin/add_household', methods=['GET', 'POST'])
def adminAddHousehold():
    user_id = request.args.get('user_id')
    form = HouseholdForm()

    household = Household.query.filter_by(user_id=user_id).first()
    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('adminUserProfiles'))

    barangay_id = user.barangay_id
    if not barangay_id:
        flash('Barangay information is missing for this user.', 'warning')
        return redirect(url_for('adminUserProfiles'))

    if household:
        form.address.data = household.address
        form.mother_first_name.data = household.mother_first_name
        form.mother_last_name.data = household.mother_last_name
        form.father_first_name.data = household.father_first_name
        form.father_last_name.data = household.father_last_name
        form.mother_birthdate.data = date.today().replace(year=date.today().year - household.mother_age)
        form.father_birthdate.data = date.today().replace(year=date.today().year - household.father_age)
    else:
        form.address.data = user.barangay.name if user.barangay else ''

    if form.validate_on_submit():
        def calculate_age(birthdate):
            today = date.today()
            return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

        mother_age = calculate_age(form.mother_birthdate.data)
        father_age = calculate_age(form.father_birthdate.data)

        try:
            if household:
                household.address = form.address.data
                household.mother_first_name = form.mother_first_name.data
                household.mother_last_name = form.mother_last_name.data
                household.father_first_name = form.father_first_name.data
                household.father_last_name = form.father_last_name.data
                household.mother_age = mother_age
                household.father_age = father_age
                household.barangay_id = barangay_id
            else:
                household = Household(
                    user_id=user_id,
                    address=form.address.data,
                    mother_first_name=form.mother_first_name.data,
                    mother_last_name=form.mother_last_name.data,
                    father_first_name=form.father_first_name.data,
                    father_last_name=form.father_last_name.data,
                    mother_age=mother_age,
                    father_age=father_age,
                    barangay_id=barangay_id
                )
                db.session.add(household)
            
            db.session.commit()
            flash('Household information saved successfully.', 'success')
            
            return redirect(url_for('add_child', user_id=user_id))
        
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error saving household: {e}")
            flash('An error occurred while saving the household information.', 'danger')

    # Pass the current date to the template
    return render_template('admin/add_household.html', form=form, user=user, today=date.today())

# bhw_manage_child Route
@app.route('/bhwManageChild', methods=['GET'])
def bhw_manage_child():
    user_id = request.args.get('user_id', type=int)
    admin_id = session.get('admin_id')

    if not admin_id:
        flash("Admin login required.", "danger")
        return redirect(url_for('adminIndex'))

    # Retrieve the admin and validate if they are a BHW or RHU
    admin = Admin.query.get(admin_id)
    if not admin:
        flash("Admin account does not exist. Please contact support.", "danger")
        return redirect(url_for('adminDashboard'))

    # Ensure the user exists
    user = User.query.get(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("adminUserProfiles"))

    # Ensure the user belongs to the admin's assigned barangay
    if user.barangay_id != admin.barangay.id:  # Use `admin.barangay.id`
        flash("You do not have permission to manage this user's children.", "danger")
        return redirect(url_for("adminUserProfiles"))

    # Retrieve only the children and households associated with the specified user
    children_with_households = (
        db.session.query(Child, Household)
        .join(Household, Household.user_id == Child.user_id)
        .filter(Child.user_id == user_id)
        .all()
    )

    # Retrieve the latest predictions only for this user’s children
    latest_predictions = {
        child.id: PredictionData.query
            .filter_by(child_id=child.id)
            .order_by(PredictionData.prediction_date.desc())
            .first()
        for child, _ in children_with_households
    }

    children = [child for child, _ in children_with_households]

    return render_template(
        'admin/manage_child.html',
        children_with_households=children_with_households,
        latest_predictions=latest_predictions,
        user=user,
        children=children
    )


@app.route('/bhwManageChild/add', methods=['GET', 'POST'])
def add_child():
    if 'admin_id' not in session:
        flash("Please log in as an admin to add children.", "danger")
        return redirect(url_for("adminIndex"))
    
    admin_id = session['admin_id']
    admin = Admin.query.get(admin_id)
    
    if not admin:
        flash("Admin account not found. Please contact support.", "danger")
        return redirect(url_for("adminIndex"))

    # Retrieve `user_id` from the request
    user_id = request.args.get('user_id', type=int) or request.form.get('user_id', type=int)
    if not user_id:
        flash("User ID is missing. Please try again.", "danger")
        return redirect(url_for("adminUserProfiles"))

    # Ensure the user belongs to the admin's assigned barangay
    user = User.query.filter(User.id == user_id, User.barangay_id == admin.barangay.id).first()  # Accessing `admin.barangay.id` now
    
    if not user:
        flash("The specified user does not exist or is not in your assigned barangay.", "danger")
        return redirect(url_for("adminUserProfiles"))

    # Fetch the father’s last name from the Household model
    household = Household.query.filter_by(user_id=user.id).first()
    father_last_name = household.father_last_name if household else ""

    form = ChildForm()
    
    # Pre-fill the father's last name if it exists
    form.last_name.data = father_last_name

    if form.validate_on_submit():
        try:
            # Create the new child instance, setting parent_id to the user's ID
            new_child = Child(
                user_id=user.id,
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                barangay_id=user.barangay_id,
                parent_id=user.id 
            )

            # Add and commit the child to the database
            db.session.add(new_child)
            db.session.commit()
            
            flash("Child added successfully.", "success")
            return redirect(url_for("adminPredict", child_id=new_child.id))
        
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error adding child for user {user_id}: {e}")
            flash("An error occurred while adding the child. Please try again.", "danger")
    
    elif request.method == 'POST':
        app.logger.warning(f"Form validation failed with errors: {form.errors}")
        flash("Please correct the errors in the form.", "warning")

    return render_template("admin/add_child.html", form=form)


@app.route('/edit_child/<int:child_id>', methods=['GET', 'POST'])
def edit_child(child_id):
    # Ensure the user has admin permissions
    if 'admin_id' not in session:
        flash("You need to be logged in as an admin to access this page.", "danger")
        return redirect(url_for('adminIndex'))

    # Fetch the child based on the provided child_id
    child = Child.query.get_or_404(child_id)

    # Initialize the form and populate with child data for a GET request
    form = ChildForm(obj=child)

    if request.method == 'POST' and form.validate_on_submit():
        # Update child data with form data
        child.first_name = form.first_name.data
        child.last_name = form.last_name.data

        # Save changes to the database
        try:
            db.session.commit()
            flash("Child's information has been updated successfully.", "success")
            return redirect(url_for('bhw_manage_child', user_id=child.user_id))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred while updating the child's information.", "danger")
            print(f"Error: {e}")

    # Render the edit form with existing child information for a GET request
    return render_template('admin/edit_child.html', form=form, child=child)


from datetime import date, timedelta

@app.route('/admin/edit_prediction/<int:prediction_id>', methods=['GET', 'POST'])
def edit_prediction(prediction_id):
    if 'admin_id' not in session:
        flash("Please log in as an admin to add children.", "danger")
        return redirect(url_for("adminIndex"))

    # Retrieve the prediction to edit
    prediction = PredictionData.query.get_or_404(prediction_id)

    # Retrieve the associated child and household for the prediction
    child = Child.query.get(prediction.child_id)
    household = Household.query.filter_by(user_id=child.user_id).first()

    # Initialize the form with the prediction's data
    form = PredictionForm(obj=prediction)
    
    # Convert the age in months back to an approximate birthdate for display in the form
    if prediction.age is not None:
        months_ago = prediction.age
        approximate_birthdate = date.today() - timedelta(days=30 * months_ago)  # Approximate month length as 30 days
        form.age.data = approximate_birthdate  # Set the date field

    # Pre-fill mother's age in the form if household data exists
    if household:
        form.mothers_age.data = household.mother_age

    if form.validate_on_submit():
        # Calculate age in months from the entered birthdate
        entered_birthdate = form.age.data
        age_in_months = (date.today().year - entered_birthdate.year) * 12 + (date.today().month - entered_birthdate.month)
        
        # Update prediction data and set age in months
        prediction.age = age_in_months
        form.populate_obj(prediction)
        
        db.session.commit()
        flash("Prediction updated successfully.", "success")
        return redirect(url_for('bhw_manage_child', user_id=child.user_id))

    # Render the edit prediction form with pre-filled data
    return render_template("admin/edit_prediction.html", form=form, prediction=prediction, child=child, household=household)


@app.route('/delete_predictions/<int:child_id>', methods=['POST'])
def delete_predictions_only(child_id):
    if 'admin_id' not in session:
        return jsonify({"message": "You are not logged in or do not have the required permissions."}), 403
    
    try:
        predictions = PredictionData.query.filter_by(child_id=child_id).all()
        for prediction in predictions:
            db.session.delete(prediction)
        db.session.commit()
        return jsonify({"message": "Predictions for the child were deleted successfully."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Error deleting predictions: {e}"}), 500

@app.route('/delete_child/<int:child_id>', methods=['POST'])
def delete_child_and_predictions(child_id):
    if 'admin_id' not in session:
        return jsonify({"message": "You are not logged in or do not have the required permissions."}), 403

    try:
        predictions = PredictionData.query.filter_by(child_id=child_id).all()
        for prediction in predictions:
            db.session.delete(prediction)
        
        child = Child.query.get(child_id)
        if child:
            db.session.delete(child)
            db.session.commit()
            return jsonify({"message": "Child and all associated predictions were deleted successfully."}), 200
        else:
            return jsonify({"message": "Child not found."}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Error deleting child and predictions: {e}"}), 500


@app.route('/viewAllChildren', methods=['GET'])
def view_all_children():
    # Ensure the admin is logged in
    admin_id = session.get('admin_id')
    if not admin_id:
        flash("Admin login required.", "danger")
        return redirect(url_for('adminIndex'))

    # Retrieve the admin and its barangay
    admin = Admin.query.get(admin_id)
    if not admin or not admin.barangay:
        flash("Admin account or barangay information is missing. Please contact support.", "danger")
        return redirect(url_for('adminDashboard'))

    # Get the barangay ID of the admin
    barangay_id = admin.barangay.id

    # Retrieve children and households filtered by the admin's barangay
    children_with_households = (
        db.session.query(Child, Household)
        .join(Household, Household.user_id == Child.user_id)
        .filter(Household.barangay_id == barangay_id)  # Filter by barangay_id
        .all()
    )

    # Retrieve the latest prediction for each child within the filtered barangay
    latest_predictions = {
        child.id: PredictionData.query
            .filter_by(child_id=child.id)
            .order_by(PredictionData.prediction_date.desc())
            .first()
        for child, _ in children_with_households
    }

    # Pass data to the template
    return render_template(
        'admin/view_all_child.html',
        children_with_households=children_with_households,
        latest_predictions=latest_predictions,
        user=None,  # Not tied to a specific user
        children=[child for child, _ in children_with_households]
    )



# Admin logout
@app.route('/admin/logout')
def adminLogout():
    session.pop('admin_id', None)
    session.pop('role', None)
    return redirect('/')


# -------------------------------RCHU Area --------------------------------
# RCHU login
@app.route('/rchu/login', methods=['GET', 'POST'])
def rchuLogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        rchu_account = RCHU.query.filter_by(username=username).first()
        if rchu_account:
            print("Found RCHU account:", rchu_account.username)
        if rchu_account and rchu_account.check_password(password):
            session['rchu_id'] = rchu_account.id
            print(f"RCHU ID set in session: {session['rchu_id']}")  
            flash('Logged in successfully as RCHU admin.', 'success')
            return redirect(url_for('rchuDashboard'))
        else:
            flash('Invalid username or password', 'danger')
            print("Login failed: invalid credentials")  
    return render_template('rchu/login.html')

@app.route('/debug/rchu')
def debug_rchu():
    accounts = RCHU.query.all()
    return f"Accounts: {[account.username for account in accounts]}"


# RCHU Dashboard Route
@app.route('/rchu/dashboard')
def rchuDashboard():
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))

    # Fetch core dashboard data
    total_users = User.query.count()
    total_bhws = Admin.query.count()
    total_children = Child.query.count()

    # Initialize malnutrition metrics
    malnutrition_metrics = {
        'Weight for Age': {'Severely Underweight': 0, 'Underweight': 0, 'Normal': 0, 'Overweight': 0},
        'Height for Age': {'Severely Stunted': 0, 'Stunted': 0, 'Normal': 0, 'Tall': 0},
        'Weight for Length/Height': {'Severely Wasted': 0, 'Wasted': 0, 'Normal': 0, 'Overweight': 0, 'Obese': 0}
    }
    age_groups = {'0-5': 0, '6-12': 0, '13-18': 0, '19-50': 0, '50+': 0}
    gender_counts = {'M': 0, 'F': 0, 'Other': 0}

    # Prepare structure for barangay-wise malnutrition metrics
    malnutrition_metrics_by_barangay = {}

    # Join PredictionData with Barangay for access to barangay name
    predictions = db.session.query(PredictionData, Barangay.name.label('barangay_name'))\
        .join(Barangay, PredictionData.barangay_id == Barangay.id)\
        .all()

    for prediction, barangay_name in predictions:
        # General malnutrition metrics
        malnutrition_metrics['Weight for Age'][prediction.weight_status or "Normal"] += 1
        malnutrition_metrics['Height for Age'][prediction.height_status or "Normal"] += 1
        malnutrition_metrics['Weight for Length/Height'][prediction.weight_length_status or "Normal"] += 1

        # Age and gender groupings
        if prediction.age <= 5:
            age_groups['0-5'] += 1
        elif 6 <= prediction.age <= 12:
            age_groups['6-12'] += 1
        elif 13 <= prediction.age <= 18:
            age_groups['13-18'] += 1
        elif 19 <= prediction.age <= 50:
            age_groups['19-50'] += 1
        else:
            age_groups['50+'] += 1

        gender = prediction.sex if prediction.sex in gender_counts else 'Other'
        gender_counts[gender] += 1

        # Barangay-specific malnutrition metrics
        if barangay_name not in malnutrition_metrics_by_barangay:
            malnutrition_metrics_by_barangay[barangay_name] = {
                'Weight for Age': {'Severely Underweight': 0, 'Underweight': 0, 'Normal': 0, 'Overweight': 0},
                'Height for Age': {'Severely Stunted': 0, 'Stunted': 0, 'Normal': 0, 'Tall': 0},
                'Weight for Length/Height': {'Severely Wasted': 0, 'Wasted': 0, 'Normal': 0, 'Overweight': 0, 'Obese': 0}
            }
        # Increment the specific malnutrition metric for the barangay
        malnutrition_metrics_by_barangay[barangay_name]['Weight for Age'][prediction.weight_status or "Normal"] += 1
        malnutrition_metrics_by_barangay[barangay_name]['Height for Age'][prediction.height_status or "Normal"] += 1
        malnutrition_metrics_by_barangay[barangay_name]['Weight for Length/Height'][prediction.weight_length_status or "Normal"] += 1

    # Query for barangay prediction counts for another chart
    barangay_predictions = (
        db.session.query(Barangay.name, func.count(PredictionData.id).label('prediction_count'))
        .outerjoin(PredictionData, Barangay.id == PredictionData.barangay_id)
        .group_by(Barangay.name)
        .order_by(Barangay.name)
        .all()
    )
    barangay_labels = [b[0] for b in barangay_predictions]
    barangay_data = [b[1] for b in barangay_predictions]

    return render_template(
        'rchu/dashboard.html',
        total_users=total_users,
        total_bhws=total_bhws,
        total_children=total_children,
        malnutrition_metrics=malnutrition_metrics,
        age_groups=age_groups,
        gender_counts=gender_counts,
        barangay_labels=barangay_labels,
        barangay_data=barangay_data,
        malnutrition_metrics_by_barangay=malnutrition_metrics_by_barangay
    )



# RCHU Manage BHWs Route
@app.route('/rchu/manage-bhws')
def rchuManageBHWs():
    if 'rchu_id' not in session:  
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))
    
    bhws = Admin.query.all() 
    return render_template('rchu/manage_bhws.html', bhws=bhws)

@app.route('/rchu/create', methods=['GET', 'POST'])
def create_rchu_account():
    form = CreateRCHUForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Create a new RCHU object and save it to the database
        new_rchu = RCHU(username=username)
        new_rchu.set_password(password)  # Assuming RCHU has set_password method for hashing
        try:
            db.session.add(new_rchu)
            db.session.commit()
            flash('RCHU account created successfully!', 'success')
            return redirect(url_for('manage_rchu'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error creating RCHU account: {e}", 'danger')
    else:
        # Add debugging output to check for errors if form does not validate
        print("Form validation errors:", form.errors)

    return render_template('rchu/create_rchu.html', form=form)


@app.route('/rchu/manage-rchu')
def manage_rchu():
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))

    rchus = RCHU.query.all()  # Fetch all RCHU accounts
    return render_template('rchu/manage_rchu.html', rchus=rchus)

@app.route('/rchu/manage-rchu/update/<int:id>', methods=['GET', 'POST'])
def update_rchu(id):
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))

    rchu = RCHU.query.get_or_404(id)
    form = UpdateRCHUForm(obj=rchu)

    if form.validate_on_submit():
        # Update RCHU username and password if provided
        rchu.username = form.username.data
        if form.password.data:
            rchu.set_password(form.password.data)

        try:
            db.session.commit()
            flash('RCHU account updated successfully!', 'success')
            return redirect(url_for('manage_rchu'))
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {e}", 'danger')
            print(f"Database commit error: {e}")

    else:
        # Print errors if form validation fails
        if form.errors:
            print("Form validation errors:", form.errors)

    return render_template('rchu/update_rchu.html', form=form, rchu=rchu)

# Delete an RCHU account
@app.route('/rchu/manage-rchu/delete/<int:id>', methods=['POST'])
def delete_rchu(id):
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))

    rchu = RCHU.query.get_or_404(id)
    db.session.delete(rchu)
    db.session.commit()
    flash('RCHU account deleted successfully!', 'success')
    return redirect(url_for('manage_rchu'))

# Add new BHW (Admin)

@app.route('/rchuCreateBHW', methods=['GET', 'POST'])
def rchuCreateBHW():
    form = CreateAdminForm()  

    if request.method == 'POST' and form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        barangay_name = form.barangay.data

        # Retrieve the Barangay object by name
        barangay = Barangay.query.filter_by(name=barangay_name).first()
        if not barangay:
            flash("Barangay not found", "danger")
            return redirect(url_for('manage_bhw')) 

        # Create the BHW account with barangay_id set to the barangay’s ID
        new_bhw = Admin(username=username, password_hash=generate_password_hash(password), barangay_id=barangay.id)

        db.session.add(new_bhw)
        try:
            db.session.commit()
            flash("BHW created successfully", "success")
        except IntegrityError:
            db.session.rollback()
            flash("Error creating BHW. Please try again.", "danger")
        
        return redirect(url_for('rchuDashboard')) 

    return render_template('rchu/create_bhw.html', form=form)  



# Update BHW
@app.route('/rchu/manage-bhws/update/<int:id>', methods=['GET', 'POST'])
def rchuUpdateBHW(id):
    if 'rchu_id' not in session: 
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('rchuLogin'))

    # Fetch the BHW admin record by ID
    bhw = Admin.query.get_or_404(id)
    
    # Initialize the form with the BHW data
    form = UpdateBHWForm(obj=bhw)
    
    # Populate the barangay dropdown choices with all available barangays
    form.barangay.choices = [(b.id, b.name) for b in Barangay.query.all()]

    # Handle form submission
    if form.validate_on_submit():
        # Update username and password hash if a new password is provided
        bhw.username = form.username.data
        if form.password.data:
            bhw.password_hash = generate_password_hash(form.password.data)
        
        # Update barangay_id with the selected barangay
        bhw.barangay_id = form.barangay.data

        db.session.commit()
        flash('BHW account updated successfully!')
        return redirect(url_for('rchuManageBHWs'))

    return render_template('rchu/update_bhw.html', form=form, bhw=bhw)



# Delete BHW
@app.route('/rchu/manage-bhws/delete/<int:bhw_id>', methods=['POST'])
def rchuDeleteBHW(bhw_id):
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('rchuLogin'))
    
    bhw = Admin.query.get_or_404(bhw_id)  
    db.session.delete(bhw)
    db.session.commit() 
    flash('BHW deleted successfully!')
    return redirect(url_for('rchuManageBHWs'))


# RCHU View All Predictions Route
@app.route('/rchu/view-predictions')
def rchuViewAllPredictions():
    if 'rchu_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('rchuLogin'))

    # Query all prediction data, ordered by barangay name
    predictions = db.session.query(
        PredictionData.id,
        PredictionData.child_first_name,
        PredictionData.child_last_name,
        PredictionData.age,
        PredictionData.prediction_result,
        PredictionData.sex,
        PredictionData.prediction_date,
        Household.father_first_name,
        Household.father_last_name,
        Barangay.name.label('barangay_name')
    ).join(Household, PredictionData.household_id == Household.id)\
     .join(Barangay, PredictionData.barangay_id == Barangay.id)\
     .order_by(Barangay.name)\
     .all()

    # Render the template with the retrieved predictions data
    return render_template('rchu/view_predictions.html', predictions=predictions)


# Route to view prediction details on the RCHU side
@app.route('/rchu/viewPrediction/<int:prediction_id>')
def rchuViewPrediction(prediction_id):
    # Check if the user has RCHU-level access
    if not session.get('rchu_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('rchuLogin'))

    # Fetch the prediction data
    prediction = PredictionData.query.get(prediction_id)

    # Redirect if prediction is not found
    if not prediction:
        flash('Prediction not found', 'danger')
        return redirect(url_for('rchuDashboard'))

    # Safely fetch associated household and child information if available
    household = Household.query.get(prediction.household_id) if prediction.household_id else None
    child = Child.query.get(prediction.child_id) if prediction.child_id else None

    # Calculate child’s age in months and determine the risk level for meal plan generation
    child_age_in_months = prediction.age
    risk_level = prediction.prediction_result

    # Access malnutrition metrics from PredictionData model
    height_status = prediction.height_status if prediction else None
    weight_status = prediction.weight_status if prediction else None
    weight_length_status = prediction.weight_length_status if prediction else None

    # Generate the meal plan
    meal_plan = generate_meal_plan(child_age_in_months, risk_level, height_status, weight_length_status)

    # Render a dedicated template for viewing a single prediction
    return render_template(
        'rchu/results.html',
        prediction=prediction,
        household=household,
        child=child,
        meal_plan=meal_plan
    )

# RCHU logout
@app.route('/rchu/logout')
def rchuLogout():
    session.pop('rchu_id', None)
    session.pop('role', None)
    return redirect(url_for('index'))

# Assign Barangay to BHW
def assign_barangay_to_admin(admin, barangay_name):
    # Query the Barangay model to find the barangay by name
    barangay = Barangay.query.filter_by(name=barangay_name).first()
    
    if barangay:
        # Append the barangay to the admin's barangays list
        admin.barangays.append(barangay)
        db.session.commit()
    else:
        raise ValueError(f"Barangay '{barangay_name}' not found.")

def initialize_rchu_account():
    """Create a default RCHU account if none exists."""
    # Check if there's already an RCHU account
    if not RCHU.query.first():
        default_rchu = RCHU(username="default_admin")
        
        # Set the password and hash it
        default_rchu.set_password("default_password") 

        print("Setting up RCHU account with hashed password:", default_rchu.password_hash)

        db.session.add(default_rchu)
        db.session.commit()
        print("Default RCHU account created successfully.")
    else:
        print("RCHU account already exists.")

# Download Predictions as CSV
@app.route('/download-predictions', defaults={'barangay_id': None})
@app.route('/download-predictions/<int:barangay_id>')
def download_predictions(barangay_id):
    # Build query to retrieve all relevant prediction data
    query = db.session.query(
        Barangay.name.label('barangay_name'),
        PredictionData.child_first_name,
        PredictionData.child_last_name,
        PredictionData.age,
        PredictionData.sex,
        PredictionData.prediction_date,
        PredictionData.weight_status,
        PredictionData.height_status,
        PredictionData.weight_length_status,
        PredictionData.prediction_result,
        Household.father_first_name,
        Household.father_last_name,
        Household.mother_first_name,
        Household.mother_last_name,
        PredictionData.vitamin_a,
        PredictionData.birth_order,
        PredictionData.breastfeeding,
        PredictionData.comorbidity_status,
        PredictionData.type_of_birth,
        PredictionData.size_at_birth,
        PredictionData.dietary_diversity_score,
        PredictionData.mothers_age,
        PredictionData.mothers_education_level,
        PredictionData.fathers_education_level,
        PredictionData.womens_autonomy_tertiles,
        PredictionData.toilet_facility,
        PredictionData.source_of_drinking_water,
        PredictionData.bmi_of_mother,
        PredictionData.number_of_children_under_five,
        PredictionData.mothers_working_status,
        PredictionData.household_size,
        PredictionData.wealth_quantile 
    ).join(Household, PredictionData.household_id == Household.id)\
     .join(Barangay, PredictionData.barangay_id == Barangay.id)
    
    # Apply filter if barangay_id is specified
    if barangay_id:
        query = query.filter(PredictionData.barangay_id == barangay_id)
    
    # Sort results by barangay name
    predictions = query.order_by(Barangay.name).all()

    # Create CSV in-memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write CSV headers with unified name columns
    writer.writerow([
        'Prediction Date', 'Barangay Name', "Child's Name", 'Age', 'Sex', 'Weight Status', 'Height Status',
        'Weight-Length Status', 'Overall Prediction Result', "Father's Name", 
        "Mother's Name", 'Vitamin A', 'Birth Order', 'Breastfeeding', 'Comorbidity Status', 
        'Type of Birth', 'Size at Birth', 'Dietary Diversity Score', "Mother's Age", 
        "Mother's Education Level", "Father's Education Level", "Women's Autonomy Tertiles", 
        'Toilet Facility', 'Source of Drinking Water', "BMI of Mother", 
        "Number of Children Under Five", "Mother's Working Status", 'Household Size', 'Wealth Quantile'
    ])

    # Write each prediction row with unified name fields
    for pred in predictions:
        # Concatenate names for unified columns
        child_name = f"{pred.child_first_name} {pred.child_last_name}"
        father_name = f"{pred.father_first_name} {pred.father_last_name}"
        mother_name = f"{pred.mother_first_name} {pred.mother_last_name}"

        writer.writerow([
            pred.prediction_date.strftime('%Y-%m-%d') if pred.prediction_date else '', pred.barangay_name,
            child_name, pred.age, pred.sex, pred.weight_status, pred.height_status,
            pred.weight_length_status, pred.prediction_result, father_name,
            mother_name, pred.vitamin_a, pred.birth_order, pred.breastfeeding, 
            pred.comorbidity_status, pred.type_of_birth, pred.size_at_birth,
            pred.dietary_diversity_score, pred.mothers_age, pred.mothers_education_level,
            pred.fathers_education_level, pred.womens_autonomy_tertiles, pred.toilet_facility,
            pred.source_of_drinking_water, pred.bmi_of_mother, pred.number_of_children_under_five,
            pred.mothers_working_status, pred.household_size, pred.wealth_quantile 
        ])

    # Set up the CSV response
    output.seek(0)  # Rewind the StringIO object
    filename = f"predictions_{'barangay_' + str(barangay_id) if barangay_id else 'all'}.csv"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }
    return Response(output, mimetype="text/csv", headers=headers)
#--------------------------------user content--------------------------------
# User login
@app.route('/user/', methods=['GET', 'POST'])
def userIndex():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('userDashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('userIndex'))

    return render_template('System4/login.html')


# User dashboard
@app.route('/user/dashboard')
def userDashboard():
    if not session.get('user_id'):
        return redirect('/user/')

    user_id = session.get('user_id')
    user = User.query.get(user_id)

    # Fetch household related to the user
    household = Household.query.filter_by(user_id=user_id).first()

    # Fetch the latest prediction for each child of the user
    children = Child.query.filter_by(user_id=user_id).all()
    latest_predictions = {
        child.id: PredictionData.query
            .filter_by(child_id=child.id)
            .order_by(PredictionData.prediction_date.desc())
            .first()
        for child in children
    }

    barangay_name = user.barangay.name if user.barangay else "Unknown Barangay"

    return render_template(
        'System4/home.html',
        title="User Dashboard",
        user=user,
        household=household,
        barangay_name=barangay_name,
        latest_predictions=latest_predictions,
        children=children
    )



@app.route('/user/results/<int:prediction_id>')
def userViewResults(prediction_id):
    # Check if the user is logged in
    if 'user_id' not in session:
        flash('You are not logged in.')
        return redirect(url_for('userIndex'))

    # Get the user ID from session
    user_id = session.get('user_id')
    
    # Fetch prediction data associated with this user
    prediction = PredictionData.query.filter_by(id=prediction_id, parent_id=user_id).first()

    # If the prediction is not found or doesn't belong to the user, redirect with a message
    if not prediction:
        flash('Prediction not found or you do not have access to this prediction.')
        return redirect(url_for('userDashboard'))

    # Fetch household data associated with the prediction
    household = Household.query.get(prediction.household_id)

    # Fetch the child associated with the prediction (if any)
    child = Child.query.get(prediction.child_id) if prediction.child_id else None

    # Fetch child’s age from PredictionData and the risk level
    child_age_in_months = prediction.age
    risk_level = prediction.prediction_result

    # Access malnutrition metrics from PredictionData model
    height_status = prediction.height_status if prediction else None
    weight_status = prediction.weight_status if prediction else None
    weight_length_status = prediction.weight_length_status if prediction else None

    # Generate the meal plan with necessary data
    meal_plan = generate_meal_plan(child_age_in_months, risk_level, height_status, weight_length_status)

    # Render the results template with prediction, household, and meal plan data
    return render_template(
        'System4/results.html',
        prediction=prediction,
        household=household,
        meal_plan=meal_plan
    )



@app.route('/user/view_profile', methods=['GET'])
def viewProfile():
    user_id = session.get('user_id')
    if not user_id:
        flash('You need to be logged in to view this page.', 'danger')
        return redirect(url_for('userIndex'))

    form = HouseholdForm()
    household = Household.query.filter_by(user_id=user_id).first()
    user = User.query.get(user_id)

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('userDashboard'))

    if household:
        # Populate form fields for view-only display
        form.address.data = household.address
        form.mother_first_name.data = household.mother_first_name
        form.mother_last_name.data = household.mother_last_name
        form.father_first_name.data = household.father_first_name
        form.father_last_name.data = household.father_last_name
    else:
        form.address.data = user.barangay.name if user.barangay else ''

    return render_template('System4/view_profile.html', form=form, user=user, mother_age=household.mother_age, father_age=household.father_age)



@app.route('/user/results/history', methods=['GET'])
def userResultsHistory():
    child_id = request.args.get('child_id', type=int)
    if not child_id:
        flash("Child ID is missing or invalid.", "danger")
        return redirect(url_for('userDashboard'))

    # Fetch the child and all predictions for the child, ordered by date
    child = Child.query.get(child_id)
    if not child:
        flash("Child not found.", "danger")
        return redirect(url_for('userDashboard'))
    
    predictions = PredictionData.query.filter_by(child_id=child_id).order_by(PredictionData.prediction_date.desc()).all()

    return render_template('System4/results_history.html', child=child, predictions=predictions)


# User Logout
@app.route('/user/logout')
def userLogout():
    if not session.get('user_id'):
        return redirect('/user/')
    
    session.clear()  # Clear all session data
    flash('You have been logged out', 'success')
    return redirect('/')


# About
@app.route('/user/about')
def about():
    if not session.get('user_id'):
        flash('User not logged in', 'danger')
        return redirect(url_for('userIndex'))
    return render_template('System4/about.html', title="e-OPT")
    

# ---------------------------- Prediction Section -------------------------------------------------- #
@app.route('/admin/predict', methods=['GET', 'POST'])
def adminPredict():
    admin_id = session.get('admin_id')
    if not admin_id:
        flash('Admin login required.', 'danger')
        return redirect(url_for('adminIndex'))

    child_id = request.args.get('child_id', type=int)
    is_update = request.args.get('is_update', 'false').lower() == 'true'

    if not child_id:
        flash("Child ID is missing or invalid.", "danger")
        return redirect(url_for('bhw_manage_child'))

    # Retrieve child and household information
    child = Child.query.get(child_id)
    household = Household.query.filter_by(user_id=child.user_id).first()
    barangay_id = household.barangay_id if household else None

    # Get the latest prediction for pre-filling the form if updating
    latest_prediction = PredictionData.query.filter_by(child_id=child_id).order_by(PredictionData.prediction_date.desc()).first()
    form = PredictionForm()

    # Pre-fill form with data if updating and a prediction exists
    if latest_prediction and is_update:
        form.age.data = date.today().replace(
            year=date.today().year - (latest_prediction.age // 12),
            month=(date.today().month - (latest_prediction.age % 12) + 12) % 12 or 12
        )
        form.sex.data = latest_prediction.sex
        form.vitamin_a.data = latest_prediction.vitamin_a
        form.birth_order.data = latest_prediction.birth_order
        form.breastfeeding.data = latest_prediction.breastfeeding
        form.comorbidity_status.data = latest_prediction.comorbidity_status
        form.type_of_birth.data = latest_prediction.type_of_birth
        form.size_at_birth.data = latest_prediction.size_at_birth
        form.dietary_diversity_score.data = latest_prediction.dietary_diversity_score
        form.mothers_age.data = latest_prediction.mothers_age
        form.mothers_education_level.data = latest_prediction.mothers_education_level
        form.fathers_education_level.data = latest_prediction.fathers_education_level
        form.womens_autonomy_tertiles.data = latest_prediction.womens_autonomy_tertiles
        form.toilet_facility.data = latest_prediction.toilet_facility
        form.source_of_drinking_water.data = latest_prediction.source_of_drinking_water
        form.bmi_of_mother.data = latest_prediction.bmi_of_mother
        form.number_of_children_under_five.data = latest_prediction.number_of_children_under_five
        form.household_size.data = latest_prediction.household_size
        form.mothers_working_status.data = latest_prediction.mothers_working_status
        form.wealth_quantile.data = latest_prediction.wealth_quantile

    # Process form submission
    if form.validate_on_submit():
        today = date.today()
        age_in_months = (today.year - form.age.data.year) * 12 + today.month - form.age.data.month
        if today.day < form.age.data.day:
            age_in_months -= 1

        # Define mappings for input categorical data
        breastfeeding_mapping = {'Yes': '1', 'No': '0'} 
        size_at_birth_mapping = {'Smaller than average': '0', 'Average': '1', 'Larger than Average': '2'}
        dietary_diversity_score_mapping = {'Below Minimum': '0', 'Minimum': '1', 'Above Minimum': '2'}
        education_level_mapping = {
            'No Proper Education': '0',
            'Elementary': '1',
            'Highschool': '2', 
            'College': '3'
        }
        water_quality_mapping = {'Clean': '0', 'Contaminated': '1', 'Untested': '2'}
        bmi_of_mother_mapping = {'Underweight': '0', 'Normal': '1', 'Overweight': '2', 'Obese': '3'}
        wealth_quantile_mapping = {'Poor': '0', 'Low Income': '1', 'Low Middle Income': '2', 'Middle Income': '3', 'Upper Middle Income': '4', 'Upper Income': '5', 'Rich': '6'}

        try:
            # Prepare model input data
            data = np.array([[
                bmi_of_mother_mapping[form.bmi_of_mother.data],
                1 if form.mothers_working_status.data == 'Working' else 0,
                breastfeeding_mapping[form.breastfeeding.data],  # Using simplified breastfeeding mapping
                1 if form.vitamin_a.data == 'Yes' else 0,
                1 if form.sex.data == 'M' else 0,
                1 if form.comorbidity_status.data == 'Yes' else 0,
                form.household_size.data,
                1 if form.type_of_birth.data == 'Singleton' else 0,
                size_at_birth_mapping[form.size_at_birth.data],
                dietary_diversity_score_mapping[form.dietary_diversity_score.data],
                age_in_months,
                form.mothers_age.data,
                form.birth_order.data,
                form.number_of_children_under_five.data,
                education_level_mapping[form.mothers_education_level.data],
                education_level_mapping[form.fathers_education_level.data],
                {'Low': '0', 'Medium': '1', 'High': '2'}[form.womens_autonomy_tertiles.data],
                1 if form.toilet_facility.data == 'Improved' else 0,
                water_quality_mapping[form.source_of_drinking_water.data],
                wealth_quantile_mapping[form.wealth_quantile.data]
            ]])

            # Debugging line to verify data preparation
            print("Prepared prediction data:", data)

            # Perform the prediction
            predictions = model.predict(data)
            weight_for_age, height_for_age, weight_for_length_height = predictions[0]
            print("Raw predictions output:", predictions)

            # Define mappings for output categories to readable labels
            status_map = {
                'Weight for Age': {'0': 'Normal', '1': 'Underweight', '2': 'Overweight', '3': 'Severely Underweight'},
                'Height for Age': {'0': 'Normal', '1': 'Severely Stunted', '2': 'Stunted', '3': 'Tall'},
                'Weight for Length/Height': {'0': 'Normal', '1': 'Wasted', '2': 'Severely Wasted', '3': 'Overweight', '4': 'Obese'}
            }

            weight_status = status_map['Weight for Age'][str(weight_for_age)]
            height_status = status_map['Height for Age'][str(height_for_age)]
            weight_length_status = status_map['Weight for Length/Height'][str(weight_for_length_height)]
            print("Mapped statuses:", weight_status, height_status, weight_length_status)

            # Create a new prediction entry in the database
            new_prediction = PredictionData(
                child_id=child.id,
                child_first_name=child.first_name,
                child_last_name=child.last_name,
                age=age_in_months,
                sex=form.sex.data,
                vitamin_a=form.vitamin_a.data,
                birth_order=form.birth_order.data,
                breastfeeding=form.breastfeeding.data,
                comorbidity_status=form.comorbidity_status.data,
                type_of_birth=form.type_of_birth.data,
                size_at_birth=form.size_at_birth.data,
                dietary_diversity_score=form.dietary_diversity_score.data,
                mothers_age=form.mothers_age.data,
                mothers_education_level=form.mothers_education_level.data,
                fathers_education_level=form.fathers_education_level.data,
                womens_autonomy_tertiles=form.womens_autonomy_tertiles.data,
                toilet_facility=form.toilet_facility.data,
                source_of_drinking_water=form.source_of_drinking_water.data,
                bmi_of_mother=form.bmi_of_mother.data,
                number_of_children_under_five=form.number_of_children_under_five.data,
                household_size=form.household_size.data,
                mothers_working_status=form.mothers_working_status.data,
                wealth_quantile=form.wealth_quantile.data,
                prediction_result=f"Weight for Age: {weight_status}<br>Height for Age: {height_status}<br>Weight for Length/Height: {weight_length_status}",
                weight_status=weight_status,
                height_status=height_status,
                weight_length_status=weight_length_status,
                prediction_date=datetime.now(),
                parent_id=child.user_id,
                household_id=household.id if household else None,
                admin_id=admin_id,
                barangay_id=barangay_id
            )

            # Save the new prediction to the database
            db.session.add(new_prediction)
            db.session.commit()
            flash('Prediction saved successfully!', 'success')
            return redirect(url_for('adminResults', prediction_id=new_prediction.id))

        except IntegrityError as ie:
            db.session.rollback()
            flash('A database error occurred.', 'danger')
            print("IntegrityError:", ie)

        except Exception as e:
            flash('An error occurred during prediction.', 'danger')
            print("Exception:", e)

    return render_template('admin/predict.html', form=form, child=child, household=household, is_update=is_update)


# Admin Results
@app.route('/admin/results/<int:prediction_id>')
def adminResults(prediction_id):
    # Ensure admin is logged in
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    # Retrieve the specific prediction by ID
    prediction = PredictionData.query.get(prediction_id)
    if not prediction:
        flash('Prediction not found', 'danger')
        return redirect(url_for('bhw_manage_child'))

    # Fetch associated household and child details
    household = Household.query.get(prediction.household_id)
    child = Child.query.get(prediction.child_id)
    user_id = child.user_id if child else None  # Get the user/parent ID if child exists

    # Generate a meal plan based on the child's age and risk level
    try:
        meal_plan = generate_meal_plan(
            age_in_months=prediction.age,
            weight_status=prediction.weight_status,
            height_status=prediction.height_status,
            weight_length_status=prediction.weight_length_status
        )
    except Exception as e:
        flash("An error occurred while generating the meal plan.", "danger")
        print("Exception:", e)
        meal_plan = None

    # Render the results.html template with the fetched data and user_id/child_id
    return render_template(
        'admin/results.html',
        prediction=prediction,
        household=household,
        child=child,
        meal_plan=meal_plan,
        user_id=user_id,
        child_id=child.id if child else None
    )


@app.route('/admin/viewResults/<int:prediction_id>')
def adminViewResultsButton(prediction_id):
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    # Fetch prediction data and its associated household and child details
    prediction = PredictionData.query.get(prediction_id)
    household = Household.query.get(prediction.household_id) if prediction else None
    child = Child.query.get(prediction.child_id) if prediction else None

    if not prediction:
        flash('Prediction not found', 'danger')
        return redirect(url_for('adminDashboard'))

    # Fetch the child’s age from PredictionData and the risk level
    child_age_in_months = prediction.age
    risk_level = prediction.prediction_result

    # Access malnutrition metrics from PredictionData model
    height_status = prediction.height_status if prediction else None
    weight_status = prediction.weight_status if prediction else None
    weight_length_status = prediction.weight_length_status if prediction else None

    # Generate meal plan
    meal_plan = generate_meal_plan(child_age_in_months, risk_level, height_status, weight_length_status)

    return render_template(
        'admin/results.html',
        prediction=prediction,
        household=household,
        child=child,
        meal_plan=meal_plan
    )

@app.route('/admin/results/history', methods=['GET'])
def adminResultsHistory():
    child_id = request.args.get('child_id', type=int)
    if not child_id:
        flash("Child ID is missing or invalid.", "danger")
        return redirect(url_for('bhw_manage_child'))

    # Fetch the child and all predictions for the child, ordered by date
    child = Child.query.get(child_id)
    if not child:
        flash("Child not found.", "danger")
        return redirect(url_for('bhw_manage_child'))
    
    predictions = PredictionData.query.filter_by(child_id=child_id).order_by(PredictionData.prediction_date.desc()).all()

    return render_template('admin/results_history.html', child=child, predictions=predictions)


@app.route('/admin/dashboard', methods=['GET'])
def adminDashboard():
    admin_id = session.get('admin_id')
    if not admin_id:
        flash('Admin login required.', 'danger')
        return redirect(url_for('adminIndex'))

    # Get BHW's assigned barangay
    admin = Admin.query.get(admin_id)
    assigned_barangay = admin.barangay 

    if not assigned_barangay:
        flash('No barangay assigned to this admin.', 'danger')
        return redirect(url_for('adminIndex'))

    # Retrieve counts and prediction data for the assigned barangay
    total_users = db.session.query(User).filter_by(barangay_id=assigned_barangay.id).count()
    total_children = db.session.query(Child).join(User).filter(User.barangay_id == assigned_barangay.id).count()

    # Initialize metrics, age groups, and gender counts
    malnutrition_metrics = {
        'Weight for Age': {'Severely Underweight': 0, 'Underweight': 0, 'Normal': 0, 'Overweight': 0},
        'Height for Age': {'Severely Stunted': 0, 'Stunted': 0, 'Normal': 0, 'Tall': 0},
        'Weight for Length/Height': {'Severely Wasted': 0, 'Wasted': 0, 'Normal': 0, 'Overweight': 0, 'Obese': 0}
    }
    age_groups = {'0-5': 0, '6-12': 0, '13-18': 0, '19-50': 0, '50+': 0}
    gender_counts = {'M': 0, 'F': 0, 'Other': 0}

    # Initialize data for barangay-based total predictions
    barangay_data = {assigned_barangay.name: 0}  # Now just one barangay

    # Filter PredictionData by assigned barangay
    predictions = PredictionData.query.filter_by(barangay_id=assigned_barangay.id).all()

    for prediction in predictions:
        # Aggregate malnutrition metrics
        malnutrition_metrics['Weight for Age'][prediction.weight_status or "Normal"] += 1
        malnutrition_metrics['Height for Age'][prediction.height_status or "Normal"] += 1
        malnutrition_metrics['Weight for Length/Height'][prediction.weight_length_status or "Normal"] += 1

        # Update barangay-specific total predictions
        barangay_data[assigned_barangay.name] += 1  # Increment total predictions count

        # Age group categorization
        if prediction.age <= 5:
            age_groups['0-5'] += 1
        elif 6 <= prediction.age <= 12:
            age_groups['6-12'] += 1
        elif 13 <= prediction.age <= 18:
            age_groups['13-18'] += 1
        elif 19 <= prediction.age <= 50:
            age_groups['19-50'] += 1
        else:
            age_groups['50+'] += 1

        # Gender categorization
        gender = prediction.sex if prediction.sex in gender_counts else 'Other'
        gender_counts[gender] += 1

    return render_template(
        'admin/testDashboard.html',
        total_users=total_users,
        total_children=total_children,
        malnutrition_metrics=malnutrition_metrics,
        age_groups=age_groups,
        gender_counts=gender_counts,
        barangay_data=barangay_data 
    )



# Delete Prediction
@app.route('/admin/delete_prediction/<int:prediction_id>', methods=['POST'])
def deletePrediction(prediction_id):
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    # Retrieve the prediction data
    prediction = PredictionData.query.get_or_404(prediction_id)
    delete_child = request.form.get('delete_child') == 'true'  # Determine deletion scope
    
    try:
        # Fetch barangay info if needed for redirection
        barangay = prediction.household.address if prediction.household else None

        # If delete_child is true, delete the associated child and their predictions
        if delete_child:
            child = prediction.child
            if child:
                for pred in child.predictions:
                    db.session.delete(pred)  # Delete all related predictions
                db.session.delete(child)  # Delete the child record
                flash('Prediction and associated child data deleted successfully.', 'success')
            else:
                flash('Associated child data not found.', 'warning')
        else:
            # Only delete the selected prediction
            db.session.delete(prediction)
            flash('Prediction deleted successfully.', 'success')

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting prediction: {e}', 'danger')

    # Redirect to the appropriate page after deletion
    return redirect(url_for('adminBarangayPredictions', barangay=barangay))


# Meal Plan Generation
def generate_meal_plan(age_in_months, weight_status, height_status, weight_length_status):
    # Define meal plans based on age and malnutrition metrics
    if age_in_months <= 6:
        if weight_status == 'Severely Underweight' or height_status == 'Severely Stunted' or weight_length_status == 'Severely Wasted':
            return ["Fortified formula with iron and vitamin D. Consider more frequent feedings with close monitoring."]
        elif weight_status == 'Underweight' or height_status == 'Stunted' or weight_length_status == 'Wasted':
            return ["Breastfeeding or formula with additional vitamin D and iron supplements."]
        else:  # Normal or other statuses
            return ["Exclusive breastfeeding or infant formula as recommended."]

    elif 7 <= age_in_months <= 12:
        if weight_status == 'Severely Underweight' or height_status == 'Severely Stunted' or weight_length_status == 'Severely Wasted':
            return [
                "Breakfast: High-calorie, iron-fortified cereal with formula.",
                "Lunch: Mashed beans and fortified sweet potatoes with chicken.",
                "Snack: Soft fruits with added iron supplements.",
                "Dinner: Fortified rice porridge with beef and spinach puree."
            ]
        elif weight_status == 'Underweight' or height_status == 'Stunted' or weight_length_status == 'Wasted':
            return [
                "Breakfast: Fortified cereal with formula.",
                "Lunch: Mashed lentils, carrots, and chicken.",
                "Snack: Iron-rich fruits like pureed apricot.",
                "Dinner: Rice with spinach puree and mashed meat."
            ]
        else:  # Normal or other statuses
            return [
                "Breakfast: Mashed banana with formula.",
                "Lunch: Soft-cooked vegetables with chicken.",
                "Snack: Soft fruits like mango puree.",
                "Dinner: Rice cereal with mashed carrots."
            ]

    elif 13 <= age_in_months <= 24:
        if weight_status == 'Severely Underweight' or height_status == 'Severely Stunted' or weight_length_status == 'Severely Wasted':
            return [
                "Breakfast: Fortified oatmeal with eggs and whole milk.",
                "Lunch: High-protein chicken stew with rice.",
                "Snack: Full-fat yogurt with fruit.",
                "Dinner: Protein-rich foods like beef stew with beans and vegetables."
            ]
        elif weight_status == 'Underweight' or height_status == 'Stunted' or weight_length_status == 'Wasted':
            return [
                "Breakfast: Scrambled eggs with whole wheat toast and milk.",
                "Lunch: Fish with rice and green peas.",
                "Snack: Fortified yogurt with fruits.",
                "Dinner: Lentil soup with chicken and mixed vegetables."
            ]
        else:  # Normal or other statuses
            return [
                "Breakfast: Oatmeal with mashed banana.",
                "Lunch: Chicken with rice and steamed vegetables.",
                "Snack: Yogurt with fruit.",
                "Dinner: Vegetable stew with lean beef and potatoes."
            ]

    elif 25 <= age_in_months <= 60:  # Preschool age (up to 5 years)
        if weight_status == 'Severely Underweight' or height_status == 'Severely Stunted' or weight_length_status == 'Severely Wasted':
            return [
                "Breakfast: Iron-fortified cereal with whole milk.",
                "Lunch: High-protein lentil and rice stew with vegetables.",
                "Snack: Whole milk with peanut butter toast.",
                "Dinner: Mashed beans with lean beef and spinach."
            ]
        elif weight_status == 'Underweight' or height_status == 'Stunted' or weight_length_status == 'Wasted':
            return [
                "Breakfast: Whole-grain cereal with milk and fruit.",
                "Lunch: Fish with rice and spinach.",
                "Snack: Peanut butter sandwich on whole grain bread.",
                "Dinner: Beef and vegetable stew with lentils."
            ]
        else:  # Normal or other statuses
            return [
                "Breakfast: Scrambled eggs with toast and milk.",
                "Lunch: Grilled chicken with rice and steamed broccoli.",
                "Snack: Apple slices with cheese.",
                "Dinner: Vegetable stew with lean pork."
            ]

    else:
        return ["Meal plan not available for this age range."]

        
with app.app_context():
    initialize_rchu_account()
        
if __name__ == '__main__':
    app.run(debug=True)
