async function runDetection() {

  // fake network data (must match your columns)
  const networkData = {};
  
  // NOTE: replace with real column names later
  for (let i = 0; i < 41; i++) {
    networkData["f" + i] = Math.random();
  }

  // fake process data (10 timesteps)
  const processData = [];
  for (let i = 0; i < 10; i++) {
    processData.push([Math.random(), Math.random(), Math.random()]);
  }

  const res = await fetch("/detect", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      network: networkData,
      process: processData
    })
  });

  const data = await res.json();

  document.getElementById("result").innerText =
    "Network: " + data.network + " | Process: " + data.process;
}