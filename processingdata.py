import pandas as pd
import os
from pathlib import Path
from flask import request


# === CÁC CỘT CẦN THIẾT === #
col_12 = ['Num', 'Code', 'Name', 'Birth', 'Maths', 'Literature', 'Physics', 'Chemistry', 'Biology', 'IT', 'History', 'Geography',
          'English', 'Civic Education', 'Technology', 'Physical education', 'Defense Education', 'Final grade', 'Academic ability',
          'Conduct', 'Break (P)', 'Break (K)', 'Break (T)', 'Title', 'Rating', 'Note']

col_TN = ['Num', 'Class', 'ID_Number', 'Name', 'Gender', 'Birth', 'Place_of_Birth',
          'Ethnicity', 'Council_Code', 'Maths', 'Literature', 'Physics', 'Chemistry',
          'Biology', 'History', 'Geography', 'Civic_Education', 'Foreign_Language', 'Foreign_Language_Subject', 'Note']

col_drop_12 = ['Num', 'Code', 'IT', 'Technology', 'Physical education', 'Defense Education', 'Final grade', 'Academic ability',
               'Conduct', 'Break (P)', 'Break (K)', 'Break (T)', 'Title', 'Rating', 'Note']

col_drop_TN_XH = ['Num', 'Class', 'ID_Number', 'Gender', 'Place_of_Birth', 'Ethnicity', 'Council_Code',
                  'Physics', 'Chemistry', 'Biology', 'Note']
col_drop_TN_TN = ['Num', 'Class', 'ID_Number', 'Gender', 'Place_of_Birth', 'Ethnicity', 'Council_Code',
                  'History', 'Geography', 'Civic_Education', 'Note']


def process_new_student_data(start_year, end_year, school_name):
    # Tạo thư mục lưu dữ liệu nếu chưa tồn tại
    base_folder = Path(f"Data/{school_name}")
    base_folder.mkdir(parents=True, exist_ok=True)
    
    years = list(range(start_year, end_year + 1))
    grades = [10, 11, 12]

    # Xác định các file cần đọc
    file_keys_12 = {}
    file_keys_TN = {}

    for year in years:
        for grade in grades:
            file_keys_12[f'file_{year}_{grade}_HKI'] = f'file_{year}_{grade}_HKI'
            file_keys_12[f'file_{year}_{grade}_HKII'] = f'file_{year}_{grade}_HKII'
    
    for year in years[2:]:  
        file_keys_TN[f'file_{year}_TN'] = f'file_{year}_TN'
    
    # === ĐỌC VÀ XỬ LÝ DỮ LIỆU HỌC KỲ === #
    df_12_sem = {}
    for key, file_key in file_keys_12.items():
        try:
            file_obj = request.files.get(file_key)
            if file_obj is None:
                print(f"Không tìm thấy file {file_key}")
                continue
                
            xlsx = pd.ExcelFile(file_obj)
            sheet_names = xlsx.sheet_names
            
            # Đọc và ghép các sheet
            sheet_dfs = []
            for sheet in sheet_names:
                try:
                    sheet_df = pd.read_excel(file_obj, header=None, sheet_name=sheet, names=col_12)
                    sheet_df = sheet_df.drop(col_drop_12, axis=1)
                    sheet_dfs.append(sheet_df)
                except Exception as e:
                    print(f"Lỗi khi đọc sheet {sheet} trong file {file_key}: {e}")
            
            if not sheet_dfs:
                continue
                
            df_12_sem[key] = pd.concat(sheet_dfs, ignore_index=True)
            
            # Chuẩn hóa dữ liệu
            df_12_sem[key]["Name"] = df_12_sem[key]["Name"].fillna("")
            df_12_sem[key]["Birth"] = df_12_sem[key]["Birth"].fillna("")
            df_12_sem[key]["name&birth"] = df_12_sem[key]["Name"].str.lower() + "_" + df_12_sem[key]["Birth"].astype(str)
            df_12_sem[key] = df_12_sem[key].drop(['Name', 'Birth'], axis=1)
            
        except Exception as e:
            print(f"Lỗi khi đọc {key}: {e}")

    # === GHÉP HK1 & HK2 === #
    df_12_grade = {}
    for year in years:
        for grade in grades:
            hk1_key = f'file_{year}_{grade}_HKI'
            hk2_key = f'file_{year}_{grade}_HKII'
            merged_key = f"file_{grade}_{year}"

            if hk1_key in df_12_sem and hk2_key in df_12_sem:
                try:
                    df_12_grade[merged_key] = pd.merge(
                        df_12_sem[hk1_key], 
                        df_12_sem[hk2_key], 
                        on="name&birth", 
                        suffixes=(f'_1_{grade}', f'_2_{grade}')
                    )
                except Exception as e:
                    print(f"Lỗi khi ghép {hk1_key} và {hk2_key}: {e}")

    # === GHÉP 3 NĂM THÀNH 1 BẢN GHI === #
    df_12_year = {}
    for year in years[:-2]:  # Loại bỏ 2 năm cuối vì không đủ dữ liệu 3 năm liên tiếp
        g10_key = f"file_10_{year}"
        g11_key = f"file_11_{year + 1}"
        g12_key = f"file_12_{year + 2}"
        merged_key_year = f"file_12_{year}"
        
        if g10_key in df_12_grade and g11_key in df_12_grade and g12_key in df_12_grade:
            try:
                # Ghép dữ liệu lớp 10 và 11
                temp_merge = pd.merge(
                    df_12_grade[g10_key], 
                    df_12_grade[g11_key], 
                    on="name&birth", 
                    suffixes=(f'_10_{year}', f'_11_{year+1}')
                )
                
                # Ghép với dữ liệu lớp 12
                df_12_year[merged_key_year] = pd.merge(
                    temp_merge,
                    df_12_grade[g12_key], 
                    on="name&birth"
                )
                
                # Lưu file
                output_path = base_folder / f'10_11_12_{year}.xlsx'
                df_12_year[merged_key_year].to_excel(output_path, index=False)
                
            except Exception as e:
                print(f"Lỗi khi ghép dữ liệu 3 năm cho khóa {year}: {e}")

    # === GHÉP TOÀN BỘ DỮ LIỆU HỌC TẬP === #
    df_12_all = pd.DataFrame()
    for year in years[:-2]:
        key = f"file_12_{year}"
        if key in df_12_year:
            try:
                # Loại bỏ dòng có giá trị NaN
                df_12_year[key] = df_12_year[key].dropna()
                
                # Ghép vào dataframe tổng
                if df_12_all.empty:
                    df_12_all = df_12_year[key].copy()
                else:
                    df_12_all = pd.concat([df_12_all, df_12_year[key]], ignore_index=True)
            except Exception as e:
                print(f"Lỗi khi ghép dữ liệu vào df_12 cho năm {year}: {e}")
    
    # Lưu file tổng hợp các khóa
    if not df_12_all.empty:
        output_path = base_folder / '10_11_12.xlsx'
        df_12_all.to_excel(output_path, index=False)
    
    # === XỬ LÝ DỮ LIỆU THI TỐT NGHIỆP === #
    df_TN_year = {}
    df_TN_TN_year = {}
    df_TN_XH_year = {}
    
    for key, file_key in file_keys_TN.items():
        try:
            file_obj = request.files.get(file_key)
            if file_obj is None:
                print(f"Không tìm thấy file {file_key}")
                continue
                
            xlsx = pd.ExcelFile(file_obj)
            sheet_names = xlsx.sheet_names
            
            for sheet in sheet_names:
                # Xác định dòng bắt đầu dữ liệu
                preview = pd.read_excel(file_obj, sheet_name=sheet, nrows=10, header=None)
                skiprows = next((i for i, row in preview.iterrows() if "Số thứ tự" in str(row.values)), None)
                
                if skiprows is not None:
                    try:
                        # Đọc dữ liệu từ dòng xác định
                        skiprows = skiprows + 1
                        df_TN = pd.read_excel(file_obj, header=None, sheet_name=sheet, skiprows=skiprows, names=col_TN)
                        
                        # Tách dữ liệu khối Tự nhiên và Xã hội
                        df_TN_TN_curr = df_TN.drop(col_drop_TN_TN, axis=1).copy()
                        df_TN_XH_curr = df_TN.drop(col_drop_TN_XH, axis=1).copy()
                        
                        # Chuẩn hóa dữ liệu
                        for df in [df_TN_TN_curr, df_TN_XH_curr]:
                            df["Name"] = df["Name"].fillna("")
                            df["Birth"] = df["Birth"].fillna("")
                            df["name&birth"] = df["Name"].str.lower() + "_" + df["Birth"].astype(str)
                            df = df.drop(['Name', 'Birth'], axis=1)
                        
                        df_TN_TN_year[key] = df_TN_TN_curr
                        df_TN_XH_year[key] = df_TN_XH_curr
                        break
                    except Exception as e:
                        print(f"Lỗi khi xử lý sheet {sheet} trong file tốt nghiệp {file_key}: {e}")
            
            # Loại bỏ dòng có giá trị NaN
            if key in df_TN_TN_year:
                df_TN_TN_year[key] = df_TN_TN_year[key].dropna().reset_index(drop=True)
            if key in df_TN_XH_year:
                df_TN_XH_year[key] = df_TN_XH_year[key].dropna().reset_index(drop=True)
                
        except Exception as e:
            print(f"Lỗi khi xử lý file tốt nghiệp {key}: {e}")

    # === GỘP DỮ LIỆU 12 VÀ DỮ LIỆU TỐT NGHIỆP === #
    for year in years[:-2]:
        merged_key_12 = f"file_12_{year}"
        merged_key_TN = f"file_TN_{year + 2}"
        
        if merged_key_12 in df_12_year and merged_key_TN in df_TN_TN_year:
            try:
                # Ghép dữ liệu khối tự nhiên
                df_merged_TN = pd.merge(
                    df_12_year[merged_key_12], 
                    df_TN_TN_year[merged_key_TN], 
                    on="name&birth", 
                    suffixes=(f'_12_{year}', f'_TN_{year+2}')
                )
                output_path = base_folder / f'TN_TN_{year}.xlsx'
                df_merged_TN.to_excel(output_path, index=False)
            except Exception as e:
                print(f"Lỗi khi ghép dữ liệu TN cho năm {year}: {e}")
                
        if merged_key_12 in df_12_year and merged_key_TN in df_TN_XH_year:
            try:
                # Ghép dữ liệu khối xã hội
                df_merged_XH = pd.merge(
                    df_12_year[merged_key_12], 
                    df_TN_XH_year[merged_key_TN], 
                    on="name&birth", 
                    suffixes=(f'_12_{year}', f'_XH_{year+2}')
                )
                output_path = base_folder / f'TN_XH_{year}.xlsx'
                df_merged_XH.to_excel(output_path, index=False)
            except Exception as e:
                print(f"Lỗi khi ghép dữ liệu XH cho năm {year}: {e}")

    # === GHÉP DỮ LIỆU TỐT NGHIỆP TẤT CẢ CÁC NĂM === #
    df_TN_TN_all = pd.DataFrame()
    df_TN_XH_all = pd.DataFrame()
    
    for key in df_TN_TN_year.keys():
        try:
            if df_TN_TN_all.empty:
                df_TN_TN_all = df_TN_TN_year[key].copy()
            else:
                df_TN_TN_all = pd.concat([df_TN_TN_all, df_TN_TN_year[key]], ignore_index=True)
        except Exception as e:
            print(f"Lỗi khi ghép dữ liệu TN vào df_TN_TN_all: {e}")
    
    for key in df_TN_XH_year.keys():
        try:
            if df_TN_XH_all.empty:
                df_TN_XH_all = df_TN_XH_year[key].copy()
            else:
                df_TN_XH_all = pd.concat([df_TN_XH_all, df_TN_XH_year[key]], ignore_index=True)
        except Exception as e:
            print(f"Lỗi khi ghép dữ liệu XH vào df_TN_XH_all: {e}")
    
    # Lưu file tổng hợp tốt nghiệp
    if not df_TN_TN_all.empty:
        output_path = base_folder / 'TN_TN.xlsx'
        df_TN_TN_all.to_excel(output_path, index=False)
        
    if not df_TN_XH_all.empty:
        output_path = base_folder / 'TN_XH.xlsx'
        df_TN_XH_all.to_excel(output_path, index=False)
        
    return True