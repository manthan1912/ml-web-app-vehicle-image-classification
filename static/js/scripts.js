$(document).ready(function() {
    // Theme switcher
    $('.theme-switch').click(function() {
        $('body').toggleClass('dark-mode');
        $('.card').toggleClass('bg-dark');
        $('.btn').toggleClass('btn-light btn-dark');
    });

    // Image preview
    $('#file').change(function() {
        const file = this.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function(event) {
                $('#image-preview').attr('src', event.target.result).show();
            };
            reader.readAsDataURL(file);
        }
    });

    $('#upload-form').on('submit', function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $('#prediction-result').html('<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>');

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            processData: false,
            cache: false,
            success: function(response) {
                $('#prediction-result').html('<strong class="text-success">Prediction:</strong> ' + response.prediction);
                updateTable();
            },
            error: function() {
                $('#prediction-result').html('<strong class="text-danger">Error:</strong> Error processing your request.');
            }
        });
    });

    function updateTable() {
        $.ajax({
            type: 'GET',
            url: '/predictions',
            success: function(predictions) {
                $('#predictions-table').empty();
                predictions.forEach(function(prediction, index) {
                    $('#predictions-table').append(
                        `<tr>
                            <td>${index + 1}</td>
                            <td><img src="${prediction.image_path}" alt="Uploaded Image" height="100"></td>
                            <td>${prediction.prediction}</td>
                            <td><button class="btn btn-danger" onclick="deletePrediction(${prediction.id})">Delete</button></td>
                        </tr>`
                    );
                });
            }
        });
    }

    window.deletePrediction = function(id) {
        $.ajax({
            type: 'POST',
            url: `/delete/${id}`,
            success: function(data) {
                updateTable(); // Refresh the table
            },
            error: function() {
                alert('Failed to delete the record.');
            }
        });
    };

    updateTable(); // Load initial data
});
