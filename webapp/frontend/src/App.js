import { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  Container, Grid, Typography, Card, CardContent, Divider, 
  Button, Chip, CircularProgress, Box 
} from '@mui/material';
import { 
  Assessment, History as HistoryIcon, GetApp, ExitToApp, 
  HealthAndSafety, Warning, Error as ErrorIcon, CheckCircle,
  Person
} from '@mui/icons-material';
import { 
  RadialBarChart, RadialBar, Legend, Tooltip, ResponsiveContainer, 
  BarChart, XAxis, YAxis, Bar 
} from 'recharts';
import PredictionForm from './PredictionForm';
import LoginForm from './LoginForm';
import HistoryTable from './HistoryTable';
import './App.css';

const defaultMetrics = [
  { name: 'AUC Global', value: 0.89, fill: '#2563eb' },
  { name: 'F1-Score', value: 0.49, fill: '#10b981' },
  { name: 'Accuracy', value: 0.84, fill: '#6366f1' },
  { name: 'Recall', value: 0.52, fill: '#f59e0b' }
];

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loginError, setLoginError] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('authToken') || null);
  const [user, setUser] = useState(localStorage.getItem('authUser') || null);

  useEffect(() => {
    axios.get('/api/metrics')
      .then((response) => setDashboard(response.data))
      .catch(() => setDashboard(null));
  }, []);

  useEffect(() => {
    if (token) {
      fetchHistory();
    }
  }, [token]);

  const fetchHistory = async () => {
    try {
      const response = await axios.get('/api/history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(response.data);
    } catch (err) {
      setHistory([]);
    }
  };

  const handleLogin = async ({ username, password }) => {
    setLoginError(null);
    try {
      const response = await axios.post('/api/login', { username, password });
      const { token: authToken, username: authUser } = response.data;
      localStorage.setItem('authToken', authToken);
      localStorage.setItem('authUser', authUser);
      setToken(authToken);
      setUser(authUser);
      fetchHistory();
    } catch (err) {
      setLoginError(err.response?.data?.error || 'Échec de connexion');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('authUser');
    setToken(null);
    setUser(null);
    setPrediction(null);
    setHistory([]);
    setError(null);
  };

  const handlePredict = async (payload) => {
    setError(null);
    setLoading(true);
    // Simulate delay for a better UX (as requested: "loader pendant la prédiction")
    await new Promise(resolve => setTimeout(resolve, 1500));
    try {
      const response = await axios.post('/api/predict', payload, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPrediction(response.data);
      fetchHistory();
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur de prédiction');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const response = await axios.get('/api/export/csv', {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `historique_${user}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError(err.response?.data?.error || 'Impossible d’exporter le CSV');
    }
  };

  const getRiskClass = (classification) => {
    if (!classification) return '';
    const c = classification.toLowerCase();
    if (c.includes('faible') || c.includes('bas')) return 'risk-low';
    if (c.includes('moyen') || c.includes('modéré')) return 'risk-medium';
    if (c.includes('élevé') || c.includes('fort')) return 'risk-high';
    return '';
  };

  const getRiskIcon = (classification) => {
    const riskClass = getRiskClass(classification);
    if (riskClass === 'risk-low') return <CheckCircle style={{ color: '#10b981' }} />;
    if (riskClass === 'risk-medium') return <Warning style={{ color: '#f59e0b' }} />;
    if (riskClass === 'risk-high') return <ErrorIcon style={{ color: '#ef4444' }} />;
    return <HealthAndSafety color="primary" />;
  };

  if (!token) {
    return (
      <Container maxWidth="sm" className="auth-container">
        <Card className="card-section auth-card animate-fade-in">
          <CardContent>
            <Box textAlign="center" mb={3}>
              <HealthAndSafety color="primary" sx={{ fontSize: 60 }} />
              <Typography variant="h4" className="title">
                PharmaGuard AI
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Plateforme de surveillance des réactions médicamenteuses
              </Typography>
            </Box>
            <Divider className="divider" />
            <LoginForm onLogin={handleLogin} loading={loading} error={loginError} />
          </CardContent>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" className="app-container animate-fade-in">
      <Grid container spacing={3} alignItems="flex-start" sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Typography variant="h3" component="h1" className="title">
            Dashboard Prédictif
          </Typography>
          <Typography variant="subtitle1" className="subtitle">
            Analyse en temps réel des risques de complications médicamenteuses.
          </Typography>
        </Grid>
        <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: { md: 'flex-end', xs: 'flex-start' }, alignItems: 'center' }}>
          <Chip 
            icon={<Person />} 
            label={`Dr. ${user}`} 
            color="primary" 
            variant="outlined" 
            sx={{ fontWeight: 600, px: 1 }}
          />
          <Button 
            variant="text" 
            startIcon={<ExitToApp />} 
            onClick={handleLogout} 
            className="logout-button"
          >
            Déconnexion
          </Button>
        </Grid>
      </Grid>

      <Grid container spacing={4}>
        <Grid item xs={12} md={7} lg={8}>
          <Card className="card-section">
            <CardContent>
              <Box display="flex" alignItems="center" mb={3}>
                <Assessment color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" fontWeight="600">Performance du Modèle</Typography>
              </Box>
              <Grid container spacing={2}>
                {(dashboard?.overview?.metrics ?? defaultMetrics).map((metric) => (
                  <Grid item xs={6} sm={3} key={metric.name}>
                    <div className="metric-card">
                      <Typography variant="caption" color="textSecondary">{metric.name}</Typography>
                      <Typography variant="h4">{(metric.value * 100).toFixed(0)}%</Typography>
                    </div>
                  </Grid>
                ))}
              </Grid>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={260}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="30%"
                    outerRadius="100%"
                    data={dashboard?.overview?.metrics ?? defaultMetrics}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar minAngle={15} background clockWise={true} dataKey="value" />
                    <Legend iconSize={10} layout="horizontal" verticalAlign="bottom" />
                    <Tooltip />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="card-section">
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <HistoryIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" fontWeight="600">Historique des Analyses</Typography>
              </Box>
              <Divider className="divider" />
              <HistoryTable data={history} />
              <Box mt={3} display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="body2" color="textSecondary">
                  Toutes les données sont cryptées et conformes aux normes de santé.
                </Typography>
                <Button 
                  variant="outlined" 
                  startIcon={<GetApp />} 
                  size="small"
                  onClick={handleExport}
                >
                  Exporter (CSV)
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5} lg={4}>
          <PredictionForm onPredict={handlePredict} loading={loading} />
          
          <Card className={`card-section result-card ${getRiskClass(prediction?.classification)} ${loading ? 'loading-blur' : ''}`}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6" fontWeight="600">Résultat d'Analyse</Typography>
                {prediction && getRiskIcon(prediction.classification)}
              </Box>
              <Divider className="divider" />
              
              {loading ? (
                <div className="loading-overlay">
                  <CircularProgress size={40} thickness={4} />
                  <Typography variant="body2" sx={{ mt: 2, fontWeight: 500 }}>
                    Calcul des probabilités cliniques...
                  </Typography>
                </div>
              ) : error ? (
                <Box textAlign="center" py={2}>
                  <ErrorIcon color="error" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography color="error">{error}</Typography>
                </Box>
              ) : prediction ? (
                <div className="animate-fade-in">
                  <Box display="flex" alignItems="center" mb={1}>
                    <span className={`risk-dot dot-${getRiskClass(prediction.classification).replace('risk-', '')}`}></span>
                    <Typography variant="h5" className="result-label">
                      {prediction.classification}
                    </Typography>
                  </Box>
                  <Typography variant="h3" className="result-value">
                    {Math.round(prediction.probability * 100)}%
                  </Typography>
                  <Typography variant="body2" paragraph color="textPrimary" sx={{ fontWeight: 500 }}>
                    {prediction.recommendation}
                  </Typography>
                  
                  <Box mt={3}>
                    <Typography variant="subtitle2" gutterBottom color="textSecondary">
                      Facteurs influents :
                    </Typography>
                    <Box display="flex" flexWrap="wrap">
                      {(prediction.explanation || []).map((factor, idx) => (
                        <span key={idx} className="feature-tag">
                          {factor}
                        </span>
                      ))}
                    </Box>
                  </Box>
                </div>
              ) : (
                <Box textAlign="center" py={4} color="textSecondary">
                  <Assessment sx={{ fontSize: 48, opacity: 0.2, mb: 2 }} />
                  <Typography variant="body2">
                    En attente de données patient pour lancer l'analyse prédictive.
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;

