import { useState } from 'react';
import { Card, CardContent, TextField, Button, Typography, Grid, MenuItem, Select, FormControl, InputLabel } from '@mui/material';
import { LocalHospital, Person, Scale, Medication, History, Announcement, Biotech } from '@mui/icons-material';

const initialValues = {
  age: '',
  sex: 'Homme',
  weight: '',
  medication: '',
  medicalHistory: 'Non',
  medicalHistoryText: '',
  allergies: 'Non',
  allergiesText: '',
  labResults: '',
};

function PredictionForm({ onPredict, loading }) {
  const [values, setValues] = useState(initialValues);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onPredict(values);
  };

  return (
    <Card className="card-section">
      <CardContent>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
          <LocalHospital color="primary" style={{ marginRight: '0.5rem' }} />
          <Typography variant="h5" style={{ fontWeight: 600 }}>Dossier Patient</Typography>
        </div>
        <Typography variant="body2" color="textSecondary" paragraph>
          Remplissez les informations cliniques pour évaluer le risque de réaction médicamenteuse.
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Âge"
                name="age"
                type="number"
                value={values.age}
                onChange={handleChange}
                required
                InputProps={{ startAdornment: <Person style={{ marginRight: 8, color: '#64748b' }} /> }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Sexe</InputLabel>
                <Select name="sex" value={values.sex} label="Sexe" onChange={handleChange}>
                  <MenuItem value="Homme">Homme</MenuItem>
                  <MenuItem value="Femme">Femme</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Poids (kg)"
                name="weight"
                type="number"
                value={values.weight}
                onChange={handleChange}
                required
                InputProps={{ startAdornment: <Scale style={{ marginRight: 8, color: '#64748b' }} /> }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Médicament"
                name="medication"
                value={values.medication}
                onChange={handleChange}
                required
                InputProps={{ startAdornment: <Medication style={{ marginRight: 8, color: '#64748b' }} /> }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Antécédents</InputLabel>
                <Select name="medicalHistory" value={values.medicalHistory} label="Antécédents" onChange={handleChange}>
                  <MenuItem value="Oui">Oui</MenuItem>
                  <MenuItem value="Non">Non</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Allergies</InputLabel>
                <Select name="allergies" value={values.allergies} label="Allergies" onChange={handleChange}>
                  <MenuItem value="Oui">Oui</MenuItem>
                  <MenuItem value="Non">Non</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {values.medicalHistory === 'Oui' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Détails antécédents"
                  name="medicalHistoryText"
                  value={values.medicalHistoryText}
                  onChange={handleChange}
                  InputProps={{ startAdornment: <History style={{ marginRight: 8, marginTop: -20, color: '#64748b' }} /> }}
                />
              </Grid>
            )}

            {values.allergies === 'Oui' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Détails allergies"
                  name="allergiesText"
                  value={values.allergiesText}
                  onChange={handleChange}
                  InputProps={{ startAdornment: <Announcement style={{ marginRight: 8, marginTop: -20, color: '#64748b' }} /> }}
                />
              </Grid>
            )}

            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Résultats biologiques (optionnel)"
                name="labResults"
                value={values.labResults}
                onChange={handleChange}
                placeholder="Ex: Taux de créatinine, enzymes hépatiques..."
                InputProps={{ startAdornment: <Biotech style={{ marginRight: 8, marginTop: -40, color: '#64748b' }} /> }}
              />
            </Grid>
          </Grid>
          <Button 
            type="submit" 
            variant="contained" 
            color="primary" 
            fullWidth 
            disabled={loading} 
            className="submit-button"
          >
            {loading ? 'Analyse en cours...' : 'Prédire le risque'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default PredictionForm;

