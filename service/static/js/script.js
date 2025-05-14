document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const area = formData.get('area');
    const rooms_count = formData.get('rooms_count');
    const floors_count = formData.get('floors_count');
    const floor = formData.get('floor');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            area: parseFloat(area),
            rooms_count: parseFloat(rooms_count),
            floors_count: parseFloat(floors_count),
            floor: parseFloat(floor)
        })
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