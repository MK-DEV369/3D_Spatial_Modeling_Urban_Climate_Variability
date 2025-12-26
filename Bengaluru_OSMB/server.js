const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(express.static('public'));
app.use('/data', express.static('data'));

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});