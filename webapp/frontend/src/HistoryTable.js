import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, Chip } from '@mui/material';

function HistoryTable({ data }) {
  if (!data?.length) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem' }}>
        <Typography variant="body2" color="textSecondary">Aucun historique disponible.</Typography>
      </div>
    );
  }

  const getChipColor = (classification) => {
    if (!classification) return 'default';
    const c = classification.toLowerCase();
    if (c.includes('faible')) return 'success';
    if (c.includes('moyen')) return 'warning';
    if (c.includes('élevé')) return 'error';
    return 'default';
  };

  return (
    <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid #e2e8f0', borderRadius: '12px', overflow: 'hidden' }}>
      <Table size="small">
        <TableHead sx={{ backgroundColor: '#f8fafc' }}>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Patient</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Médicament</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Risque</TableCell>
            <TableCell sx={{ fontWeight: 600 }} align="right">Probabilité</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, index) => (
            <TableRow key={`${row.created_at}-${index}`} hover>
              <TableCell sx={{ fontSize: '0.85rem' }}>
                {new Date(row.created_at).toLocaleDateString('fr-FR')} {new Date(row.created_at).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
              </TableCell>
              <TableCell>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {row.payload?.age} ans, {row.payload?.sex === 'Homme' ? 'H' : 'F'}
                </Typography>
              </TableCell>
              <TableCell>
                <Chip label={row.payload?.medication} size="small" variant="outlined" />
              </TableCell>
              <TableCell>
                <Chip 
                  label={row.prediction?.classification?.split(' ')[0]} 
                  size="small" 
                  color={getChipColor(row.prediction?.classification)}
                  sx={{ fontWeight: 600 }}
                />
              </TableCell>
              <TableCell align="right" sx={{ fontWeight: 700, color: '#2563eb' }}>
                {Math.round(row.prediction?.probability * 100)}%
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default HistoryTable;

