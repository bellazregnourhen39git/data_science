import { useState } from 'react';
import { TextField, Button, Typography } from '@mui/material';

const initialValues = {
  username: '',
  password: ''
};

function LoginForm({ onLogin, loading, error }) {
  const [values, setValues] = useState(initialValues);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onLogin(values);
  };

  return (
    <form onSubmit={handleSubmit}>
      <TextField
        fullWidth
        label="Nom d’utilisateur"
        name="username"
        value={values.username}
        onChange={handleChange}
        margin="normal"
      />
      <TextField
        fullWidth
        label="Mot de passe"
        type="password"
        name="password"
        value={values.password}
        onChange={handleChange}
        margin="normal"
      />
      {error && (
        <Typography color="error" variant="body2" style={{ marginTop: 12 }}>
          {error}
        </Typography>
      )}
      <Button 
        type="submit" 
        variant="contained" 
        color="primary" 
        fullWidth 
        disabled={loading} 
        className="submit-button"
      >
        {loading ? 'Connexion...' : 'Se connecter'}
      </Button>
    </form>
  );
}

export default LoginForm;
