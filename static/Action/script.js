function redirectToPage2() {
    window.location.href = "page2"; // Chuyển đến page2
}

function redirectToPage1() {
    window.location.href = "page1"; // Chuyển đến page1
}


function redirectToPage5() {
    window.location.href = "page5"; // Chuyển đến page5
}

function redirectToPage6() {
    window.location.href = "page6"; // Chuyển đến page6
}

function redirectToPageDashBoard() {
    window.location.href = "page_dashboard"; // Chuyển đến page dashboard
}

function redirectToPageAdmin() {
    window.location.href = "page_admin"; // Chuyển đến pageadmin
}

document.addEventListener('keydown', function (event) {
    const inputs = document.querySelectorAll('.input input[type="number"]');
    const inputArray = Array.from(inputs);
    const currentIndex = inputArray.indexOf(document.activeElement);
    
    const totalRows = inputs.length > 36 ? 6 : 4; // Xác định số lượng ô input trong trang hiện tại
    
    let nextIndex;

    switch (event.key) {
        case 'ArrowUp':
            nextIndex = currentIndex - totalRows; // Lên trên cùng một cột
            break;
        case 'ArrowDown':
            nextIndex = currentIndex + totalRows; // Xuống dưới cùng một cột
            break;
        case 'ArrowLeft':
            nextIndex = currentIndex - 1; // Sang trái
            break;
        case 'ArrowRight':
            nextIndex = currentIndex + 1; // Sang phải
            break;
        default:
            return; // Nếu không phải các phím điều hướng, thoát khỏi hàm
    }

    if (nextIndex >= 0 && nextIndex < inputArray.length) {
        inputArray[nextIndex].focus();
    }

    // Ngăn việc di chuyển con trỏ trong input khi nhấn phím điều hướng
    event.preventDefault();
});




function toggleCheckbox(selected) {
    const natureCheckbox = document.getElementById('nature_checkbox');
    const socialCheckbox = document.getElementById('social_checkbox');

    // Nếu "Tự nhiên" được chọn, bỏ chọn "Xã hội"
    if (selected === 'nature') {
        if (natureCheckbox.checked) {
            socialCheckbox.checked = false;
        }
    }

    // Nếu "Xã hội" được chọn, bỏ chọn "Tự nhiên"
    if (selected === 'social') {
        if (socialCheckbox.checked) {
            natureCheckbox.checked = false;
        }
    }
}



function toggleCheckbox1(selected) {
    const natureCheckbox1 = document.getElementById('nature_checkbox1');
    const socialCheckbox1 = document.getElementById('social_checkbox1');

    // Nếu "Tự nhiên" được chọn, bỏ chọn "Xã hội"
    if (selected === 'nature') {
        if (natureCheckbox1.checked) {
            socialCheckbox1.checked = false;
        }
    }

    // Nếu "Xã hội" được chọn, bỏ chọn "Tự nhiên"
    if (selected === 'social') {
        if (socialCheckbox1.checked) {
            natureCheckbox1.checked = false;
        }
    }
}







function showImage(n) {
    let dropdown, image;

    if (n == 1) {
        dropdown = document.getElementById("imageDropdown1");
        image = document.getElementById("displayedImage1");
    } else if (n == 2) {
        dropdown = document.getElementById("imageDropdown2");
        image = document.getElementById("displayedImage2");
    } else if (n == 3){
        dropdown = document.getElementById("imageDropdown3");
        image = document.getElementById("displayedImage3");
    } else {
        dropdown = document.getElementById("imageDropdown4");
        image = document.getElementById("displayedImage4");
    }
    const selectedValue = dropdown.value;

    if (selectedValue) {
        image.src = selectedValue;
        image.style.display = "block";
    } else {
        image.style.display = "none";
    }
}


function goBack() {
    window.history.back();
}



function showLayout1() {
    const layout1 = document.getElementById('layout1');
    const layout2 = document.getElementById('layout2');
    layout1.classList.add('active');
    layout2.classList.remove('active');
    localStorage.setItem('selectedLayout', 'layout1');
}

function showLayout2() {
    const layout1 = document.getElementById('layout1');
    const layout2 = document.getElementById('layout2');
    layout2.classList.add('active');
    layout1.classList.remove('active');
    localStorage.setItem('selectedLayout', 'layout2');
}

function showLayout1in3() {
    const layout1 = document.getElementById('layout1');
    const layout2 = document.getElementById('layout2');
    const layout3 = document.getElementById('layout3');
    layout1.classList.add('active');
    layout2.classList.remove('active');
    layout3.classList.remove('active');
    localStorage.setItem('selectedLayout', 'layout1');
}

function showLayout2in3() {
    const layout1 = document.getElementById('layout1');
    const layout2 = document.getElementById('layout2');
    const layout3 = document.getElementById('layout3');
    layout2.classList.add('active');
    layout1.classList.remove('active');
    layout3.classList.remove('active');
    localStorage.setItem('selectedLayout', 'layout2');
}

function showLayout3in3() {
    const layout1 = document.getElementById('layout1');
    const layout2 = document.getElementById('layout2');
    const layout3 = document.getElementById('layout3');
    layout3.classList.add('active');
    layout1.classList.remove('active');
    layout2.classList.remove('active');
    localStorage.setItem('selectedLayout', 'layout3');
}




function fetchSubmitDataFromBackend() {
    fetch('get_submit_data')
        .then(response => response.json())
        .then(data => {
            let totalSubmit = data.total_submit;
            let todaySubmit = data.today_submit;
            document.getElementById('total-submit').innerText = totalSubmit;
            document.getElementById('today-submit').innerText = todaySubmit;
        });
}


function fetchPerformanceMulti() {
    fetch('getperformance/multi/average')
    .then(response => response.json())
    .then(data => {
        document.getElementById('multi-latency').textContent = data.avg_latency + " ms";
        document.getElementById('multi-throughput').textContent = data.avg_throughput + " req/s";
        document.getElementById('multi-cpu').textContent = data.avg_cpu_usage + " %";
        document.getElementById('multi-memory').textContent = data.avg_memory_usage + " MB";
        document.getElementById('multi-total').textContent = data.avg_total_predictions;
    });
}


function fetchPerformanceSingle() {
    fetch('getperformance/single/average')
    .then(response => response.json())
    .then(data => {
        document.getElementById('single-latency').textContent = data.avg_latency + " ms";
        document.getElementById('single-throughput').textContent = data.avg_throughput + " req/s";
        document.getElementById('single-cpu').textContent = data.avg_cpu_usage + " %";
        document.getElementById('single-memory').textContent = data.avg_memory_usage + " MB";
    });
}

function toggleDropdown(event) {
    const dropdown = document.getElementById('admin-dropdown');
    dropdown.classList.toggle('hidden');
    event.stopPropagation(); 
}

document.addEventListener('click', function (e) {
    const adminMenu = document.querySelector('.admin-menu');
    const dropdown = document.getElementById('admin-dropdown');
    if (!adminMenu.contains(e.target)) {
        dropdown.classList.add('hidden');
    }
});



  