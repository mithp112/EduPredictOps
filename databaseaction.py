from database import Admin, ListSchool, PerformanceMulti, PerformanceSingle, SubmitData
from flask import jsonify
from datetime import date, datetime
from mongoengine.queryset.visitor import Q
import psutil



def verify_admin(username, password):
    admin = Admin.objects(username=username).first()
    if admin and admin.check_password(password):
        return True
    else:
        return False

def updateAdmin(username, password, new_password):
    admin = Admin.objects(username=username).first()
    if admin and admin.check_password(password):
        admin.set_password(new_password)
        admin.save()
        return True
    else:
        return False
    


def get_all_schools_and_years():
    schools = ListSchool.objects()
    return [{
        'schoolName': school.schoolName,
        'years': school.years,
        'years_in_use': school.yearsInUse
    } for school in schools]


def get_school_id_by_name(school_name):
    school = ListSchool.objects(schoolName=school_name).first()
    return str(school.id) if school else None


def addSchool(school_name, start_year, end_year):
    school = ListSchool.objects(schoolName=school_name).first()
    years_to_add = list(range(start_year, end_year + 1))

    if not school:
        new_school = ListSchool(
            schoolName=school_name,
            years=years_to_add,
            yearsInUse=years_to_add
        )
        new_school.save()
    else:
        current_years = set(school.years)
        updated_years = current_years.union(set(years_to_add))
        school.years = sorted(list(updated_years))
        school.save()


def updateSchool(school_name, listYear):
    school = ListSchool.objects(schoolName=school_name).first()
    if school:
        school.yearsInUse = sorted(listYear)
        school.save()


def log_performance_multi(school_name, performance):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        print("School not found!")
        return

    metrics = PerformanceMulti(
        timestamp=performance['timestamp'],
        latency=performance['latency'],
        throughput=performance['throughput'],
        cpu_usage=performance['cpu_usage'],
        memory_usage=performance['memory_usage'],
        total_predictions=performance['total_predictions'],
        school_id=school_id
    )
    metrics.save()


def log_performance_single(school_name, performance):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return "School not found!"

    metrics = PerformanceSingle(
        timestamp=performance['timestamp'],
        latency=performance['latency'],
        throughput=performance['throughput'],
        cpu_usage=performance['cpu_usage'],
        memory_usage=performance['memory_usage'],
        school_id=school_id
    )
    metrics.save()


def get_performance_single(school_name):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return jsonify({"error": "School not found"})

    records = PerformanceSingle.objects(school_id=school_id)
    if not records:
        return jsonify({})

    return jsonify({
        "avg_latency": records.latency,
        "avg_throughput": records.throughput,
        "avg_cpu_usage": records.cpu_usage,
        "avg_memory_usage": records.memory_usage,
    })


def get_performance_multi(school_name):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return jsonify({"error": "School not found"})

    records = PerformanceMulti.objects(school_id=school_id)
    if not records:
        return jsonify({})

    return jsonify({
        "avg_latency": records.latency,
        "avg_throughput": records.throughput,
        "avg_cpu_usage": records.cpu_usage,
        "avg_memory_usage": records.memory_usage,
        "avg_total_predictions": records.total_predictions
    })


def get_performance_single_average(school_name):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return jsonify({"error": "School not found"})

    records = PerformanceSingle.objects(school_id=school_id)
    if not records:
        return jsonify({})

    avg_latency = sum([r.latency for r in records]) / len(records)
    avg_throughput = sum([r.throughput for r in records]) / len(records)
    avg_cpu = sum([r.cpu_usage for r in records]) / len(records)
    avg_memory = sum([r.memory_usage for r in records]) / len(records)

    return jsonify({
        "avg_latency": round(avg_latency, 2),
        "avg_throughput": round(avg_throughput, 2),
        "avg_cpu_usage": round(avg_cpu, 2),
        "avg_memory_usage": round(avg_memory, 2)
    })


def get_performance_multi_average(school_name):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return jsonify({"error": "School not found"})

    records = PerformanceMulti.objects(school_id=school_id)
    if not records:
        return jsonify({})

    avg_latency = sum([r.latency for r in records]) / len(records)
    avg_throughput = sum([r.throughput for r in records]) / len(records)
    avg_cpu = sum([r.cpu_usage for r in records]) / len(records)
    avg_memory = sum([r.memory_usage for r in records]) / len(records)
    avg_predictions = sum([r.total_predictions for r in records]) / len(records)

    return jsonify({
        "avg_latency": round(avg_latency, 2),
        "avg_throughput": round(avg_throughput, 2),
        "avg_cpu_usage": round(avg_cpu, 2),
        "avg_memory_usage": round(avg_memory, 2),
        "avg_total_predictions": round(avg_predictions, 2)
    })


def log_access(endpoint, school_name):
    school_id = get_school_id_by_name(school_name)
    if school_id is None:
        return jsonify({"error": "School not found"})

    today = date.today()
    
    # Lấy bản ghi SubmitData theo school_id
    submit_data = SubmitData.objects(school_id=school_id).first()

    if not submit_data:
        submit_data = SubmitData(
            school_id=school_id,
            total_submits=1,
            submits_today=1,
            last_day_submit=today
        )
    else:
        if submit_data.last_day_submit != today:
            submit_data.submits_today = 1
            submit_data.last_day_submit = today
        else:
            submit_data.submits_today += 1
        submit_data.total_submits += 1

    submit_data.save()

  


def get_submit_data(school_name):
    try:
        school_id = get_school_id_by_name(school_name)
        if school_id is None:
            return jsonify({"error": "School not found"})

        submit_data = SubmitData.objects(school_id=school_id).first()
        today = date.today()

        if not submit_data:
            return jsonify({
                'total_submit': 0,
                'today_submit': 0
            })

        # Nếu ngày thay đổi, reset submits_today
        if submit_data.last_day_submit != today:
            submit_data.submits_today = 0
            submit_data.last_day_submit = today
            submit_data.save()

        return jsonify({
            'total_submit': submit_data.total_submits,
            'today_submit': submit_data.submits_today
        })

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': 'Internal server error'}), 500