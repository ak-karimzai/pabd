document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const area = formData.get('area');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ area: parseFloat(area) })
      });

      const data = await res.json();

      if (res.ok) {
        document.getElementById('response').textContent = `Predicted Price: ${data.price} ₽`;
      } else {
        document.getElementById('response').textContent = `Error: ${data.error || "Unknown error"}`;
      }
    } catch (err) {
      document.getElementById('response').textContent = `Network Error: ${err.message}`;
    }
});