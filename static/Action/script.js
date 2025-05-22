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

function redirectToPageAuto() {
    window.location.href = "page_auto"; // Chuyển đến page6
}

function redirectToPageDashBoard() {
    window.location.href = "page_dashboard"; // Chuyển đến page dashboard
}

function redirectToPageAdmin() {
    window.location.href = "page_admin"; // Chuyển đến pageadmin
}

function redirectToPageProcessing() {
    window.location.href = "page_processing_data"; // Chuyển đến page processing
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


async function fetchSchoolList() {
    try {
        const response = await fetch("/get_all_schools_and_years");
        if (!response.ok) throw new Error("Không thể lấy danh sách trường.");

        const schoolList = await response.json();
        const dropdown = document.getElementById("schoolListDropdown");

        // Xóa các lựa chọn cũ
        dropdown.innerHTML = `<option value="">Danh sách các trường hiện tại</option>`;        

        schoolList.forEach((school) => {
            const option = document.createElement("option");
            option.value = school.schoolName;
            option.textContent = school.schoolName;
            dropdown.appendChild(option);
        });
        return schoolList

    } catch (error) {
        console.error("Lỗi khi tải danh sách trường:", error);
        return []
    }
}


function chosenSchool() {
    const dropdown = document.getElementById("schoolListDropdown");
    const selectedValue = dropdown.value;
    const yearDisplay = document.getElementById("displayedListYear");
    const yearInUseDisplay = document.getElementById("displayedListYearInUse");
    const hiddenSchoolInputs = document.querySelectorAll('input[name="schoolName"][type="hidden"]');
    hiddenSchoolInputs.forEach(input => {
        input.value = schoolName;
    });
    if (!selectedValue) {
        yearDisplay.innerText = "";
        yearInUseDisplay.innerText = "";
        return;
    }

    const selectedSchool = cachedSchoolList.find(school => school.schoolName === selectedValue);
    if (selectedSchool && selectedSchool.years) {
        yearDisplay.innerText = "Các năm có sẵn: " + selectedSchool.years.join(", ");
        yearInUseDisplay.innerText = "Các năm được train: " + selectedSchool.years_in_use.join(", ");
    } else {
        yearDisplay.innerText = "Không có thông tin năm học.";
    }
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



function generateUploadFields() {
    const startYear = parseInt(document.getElementById("start_year").value);
    const endYear = parseInt(document.getElementById("end_year").value);
    const container = document.getElementById("upload_fields_container");
    container.innerHTML = ""; // Xóa cũ nếu có

    if (isNaN(startYear) || isNaN(endYear) || startYear > endYear) {
        alert("Vui lòng nhập năm hợp lệ!");
        return;
    }
    const semesters = ["HK1", "HK2"];
    const grades = [10, 11, 12];

    for (let year = startYear; year <= endYear; year++) {
        grades.forEach(grade => {
            semesters.forEach(sem => {
                const label = document.createElement("label");
                label.textContent = `Dữ liệu điểm năm ${year} - Khối ${grade} - ${sem}`;
                const input = document.createElement("input");
                input.type = "file";
                input.name = `file_${year}_${grade}_${sem}`;
                input.required = true;
                container.appendChild(label);
                container.appendChild(input);
                container.appendChild(document.createElement("br"));
            });
        });

        // Nếu đủ 3 năm học, thì hiển thị file tốt nghiệp
        if (year >= startYear + 2) {
            const label = document.createElement("label");
            label.textContent = `Dữ liệu điểm TN năm ${year}`;
            const input = document.createElement("input");
            input.type = "file";
            input.name = `file_${year}_graduate`;
            input.required = true;
            container.appendChild(label);
            container.appendChild(input);
            container.appendChild(document.createElement("br"));
        }
    }
}



// function generateCheckDriftFields