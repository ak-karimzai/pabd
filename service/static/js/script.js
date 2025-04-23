document.getElementById('numberForm').addEventListener('submit', async function(e) {
  e.preventDefault();

  const formData = new FormData(this);
  const numbers = {
    num1: formData.get('num1'),
    num2: formData.get('num2'),
    num3: formData.get('num3'),
    num4: formData.get('num4')
  };

  try {
    const res = await fetch('/api/numbers', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(numbers)
    });

    const data = await res.json();
    document.getElementById('response').textContent = 'Server response: ' + JSON.stringify(data);
  } catch (err) {
    document.getElementById('response').textContent = 'Error: ' + err.message;
  }
});
