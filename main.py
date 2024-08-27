# Import the create_app function from the website module
from website import create_app

# Call the create_app function to set up the Flask application
app = create_app()

# Check if this script is being run directly (not imported)
if __name__ == '__main__':
    # Run the Flask application with debug mode enabled
    # Debug mode provides detailed error messages and auto-reloads the server when code changes
    app.run(debug=True)