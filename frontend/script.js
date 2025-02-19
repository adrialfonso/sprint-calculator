const columnOptions = [
    '10m', '20m', '30m', '40m', '50m', '60m', '70m', '80m', '90m', '100m', '120m', '150m', '180m', '200m', 
    '10-20m', '20-30m', '30-40m', '40-50m', '50-60m', '60-70m', '70-80m', '80-90m', '90-100m',
    '10-30m', '20-40m', '30-50m', '40-60m', '50-70m', '60-80m', '70-90m', '80-100m', '100-120m',
    '10-40m', '20-50m', '30-60m', '40-70m', '50-80m', '60-90m', '70-100m', 
    '50-100m', '100-150m', '150-200m', '180-200m', 
    '50-150m', '100-200m'];
  
  let columnCount = 0;
  
  const targetColumnSelect = document.getElementById('targetColumn');
  columnOptions.forEach(option => {
    const optionElement = document.createElement('option');
    optionElement.value = option;
    optionElement.textContent = option;
    targetColumnSelect.appendChild(optionElement);
  });
  
  function addColumnInput() {
    const container = document.getElementById('columnInputs');
    const div = document.createElement('div');
    div.classList.add('column-input');
  
    const select = document.createElement('select');
    select.setAttribute('name', `inputColumn${columnCount}`);
    select.innerHTML = columnOptions.map(option => `<option value="${option}">${option}</option>`).join('');
    div.appendChild(select);
  
    const input = document.createElement('input');
    input.setAttribute('type', 'number');
    input.setAttribute('placeholder', 'Split time');
    input.setAttribute('step', 'any');
    input.required = true;
    div.appendChild(input);
  
    const sliderDiv = document.createElement('div');
    sliderDiv.classList.add('slider-container');
  
    const timingSelect = document.createElement('select');
    timingSelect.setAttribute('id', `timingSelect${columnCount}`);
  
    timingSelect.innerHTML = ` 
      <option value="" disabled selected>Timing</option>
      <option value="hand">Hand timing</option>
      <option value="electronic_no_rt">Electronic timing (no RT)</option>
      <option value="electronic_with_rt">Electronic timing (with RT)</option>
    `;
  
    sliderDiv.appendChild(timingSelect);
    div.appendChild(sliderDiv);
  
    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'Clean';
    deleteButton.type = 'button';
    deleteButton.onclick = () => div.remove();
    div.appendChild(deleteButton);
  
    container.appendChild(div);
    columnCount++;
  }
  
  addColumnInput();
  
  document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();
  
    const inputValues = {};
    const columnInputs = document.querySelectorAll('.column-input');
    
    columnInputs.forEach(input => {
      const column = input.querySelector('select').value;
      let value = parseFloat(input.querySelector('input').value);
      
      const timingSelect = input.querySelector('select[id^="timingSelect"]');
      const timingValue = timingSelect ? timingSelect.value : 'hand';  
  
      if (timingValue === 'hand') {
        value += 0.45;  
      } else if (timingValue === 'electronic_no_rt') {
        value += 0.2;   
      } else if (timingValue === 'electronic_with_rt') {
        
      }
      
      inputValues[column] = value;
    });
  
    const targetColumn = document.getElementById('targetColumn').value.trim();
  
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
  
    const csvPaths = ["100m_splits_data.csv", "200m_splits_data.csv"];
  
    try {
      const response = await fetch('https://sprint-calculator-backend.onrender.com/predict_multiple_csvs/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: inputValues, 
          target_column: targetColumn, 
          csv_paths: csvPaths
        })
      });
  
      const data = await response.json();
      console.log(data);
  
      if (response.ok) {
        const predictionText = `${targetColumn} result if the ${data.last_entry} is a split time from a ${targetColumn} race: ${data.prediction.toFixed(2)} seconds`;
        
        if (parseFloat(targetColumn) < parseFloat(data.last_entry)) {
          document.getElementById('predictionText').textContent = `Theoretical ${targetColumn} split time in the ${data.last_entry} race: ${data.prediction.toFixed(2)} seconds`;
        } else {
          document.getElementById('predictionText').textContent = predictionText;
        }
  
        document.getElementById('adjustedPrediction').textContent = data.adjusted_prediction.toFixed(2);
        document.getElementById('mae').textContent = data.mean_absolute_error.toFixed(2);
  
        if (data.adjusted_prediction.toFixed(2) === data.prediction.toFixed(2)) {
          document.getElementById('predictionText').style.display = 'none';
        } else {
          document.getElementById('predictionText').style.display = 'block';
        }
  
        document.getElementById('result').style.display = 'block';
      } else {
        document.getElementById('errorMessage').textContent = data.detail;
        document.getElementById('error').style.display = 'block';
      }
    } catch (error) {
      document.getElementById('errorMessage').textContent = `Server error: ${error.message}`;
      document.getElementById('error').style.display = 'block';
    }
  });
  