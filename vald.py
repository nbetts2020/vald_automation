import requests
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import time
import os
import urllib.parse
import re
from dotenv import load_dotenv

load_dotenv()

class Vald():
    def __init__(self):
        self.token_url = os.getenv("TOKEN_URL")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.tenant_id = os.getenv("TENANT_ID")

        self.modified_from_utc = os.getenv("MODIFIED_FROM_UTC")

        self.forceframes_api_url = os.getenv("FORCEFRAMES_API_URL")
        self.groupnames_api_url = os.getenv("GROUPNAMES_API_URL")
        self.profiles_api_url = os.getenv("PROFILES_API_URL")

        self.vald_master_file_path = os.getenv("VALD_MASTER_FILE_PATH")
        self.base_directory = os.getenv("VALD_FOLDER_PATH")
    
    def get_last_update(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_row = df.iloc[-1]
        test_date_utc = last_row['testDateUtc']
        last_index = last_row.name

        test_date_dt = datetime.strptime(test_date_utc, "%Y-%m-%dT%H:%M:%S.%fZ")

        updated_test_date_dt = test_date_dt + timedelta(milliseconds=1)
        updated_test_date_utc = updated_test_date_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        
        return updated_test_date_utc, last_index
    
    def sanitize_filename(self, filename):
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
        return sanitized

    def sanitize_foldername(self, foldername):
        sanitized = re.sub(r'[^a-zA-Z0-9_ -]', '_', foldername)
        return sanitized
    
    def get_access_token(self):
        auth_response = requests.post(
            self.token_url,
            auth=HTTPBasicAuth(self.client_id, self.client_secret),
            data={'grant_type': 'client_credentials'}
        )
        return auth_response.json()['access_token'] if auth_response.status_code == 200 else None

    def fetch_data(self, url, headers):
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None
    
    def get_tests(self, date_range):
        print(date_range, "date_range")
        access_token = self.get_access_token()
        if not access_token:
            print("Failed to retrieve access token")
            return

        headers = {'Authorization': f'Bearer {access_token}', 'Accept': '*/*'}
        api_url = f"{self.forceframes_api_url}?TenantId={self.tenant_id}&ModifiedFromUtc={self.modified_from_utc}&TestFromUtc={date_range[0]}&TestToUtc={date_range[1]}"

        tests_data = self.fetch_data(api_url, headers)
        if tests_data is None:
            return pd.DataFrame()
        api_url_groupnames = f"{self.groupnames_api_url}?TenantId={self.tenant_id}"
        group_data = self.fetch_data(api_url_groupnames, headers)
        id_to_name = {group['id']: group['name'] for group in group_data['groups']}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.fetch_data, f"{self.profiles_api_url}{test['athleteId']}?TenantId={self.tenant_id}", headers) for test in tests_data['tests']]
            for test, future in zip(tests_data['tests'], futures):
                try:
                    result = future.result()
                    if result is not None:
                        test['Name'] = result['givenName'].strip() + " " + result['familyName'].strip()
                        group_ids = result['groupIds']
                        group_names = [id_to_name.get(g_id, "ID not found") for g_id in group_ids]
                        test['Groups'] = '|'.join(group_names)
                    else:
                        continue
                        #return pd.DataFrame()
                except Exception as e:
                    print(f"An error occurred while processing test with athleteId {test['athleteId']}: {e}")
                    return pd.DataFrame()

        print("Data retrieval complete.")
        return pd.json_normalize(tests_data['tests'])
    
    def modify_df(self, df):
        df['ExternalId'] = ""
        df['Direction'] = ""
        df['adjusted_times'] = df['testDateUtc'].apply(parser.parse)

        df['Date UTC'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time UTC'] = df['adjusted_times'].dt.strftime('%I:%M %p')

        df.rename(columns={'device': 'Device', 'testTypeName': 'Test', 'testPositionName': 'Position', 'notes': 'Notes'}, inplace=True)
        df['Mode'] = "Bar + Frame"

        static_columns = ['L Reps', 'R Reps', 'L Max Force (N)', 'R Max Force (N)', 'Max Imbalance',
                        'L Max Ratio', 'R Max Ratio', 'L Avg Force (N)', 'R Avg Force (N)', 'Avg Imbalance',
                        'L Avg Ratio', 'R Avg Ratio', 'L Max Impulse (Ns)', 'R Max Impulse (Ns)', 'Impulse Imbalance (%)']
        for col in static_columns:
            df[col] = 0

        df = df.reset_index(drop=True)
        df = df.loc[df.index.repeat(2)].reset_index(drop=True)
        return df

    def parse_date_range(self, date_range):
        start_str, end_str = date_range.split('-')
        start_date = datetime.strptime(start_str, "%m/%d/%Y")
        end_date = datetime.strptime(end_str, "%m/%d/%Y")
        return start_date, end_date
    
    def generate_intervals(self, start_date, end_date, interval_days):
        intervals = []
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=interval_days)
            if current_end > end_date:
                current_end = end_date
            intervals.append((current_start, current_end - timedelta(seconds=1)))
            current_start = current_end
        return intervals

    def format_date_utc(self, date, is_start):
        if is_start:
            formatted = date.strftime("%Y-%m-%dT%H%%3A%M%%3A%SZ")
        else:
            end_time = date - timedelta(seconds=1)
            formatted = end_time.strftime("%Y-%m-%dT%H%%3A%M%%3A%SZ")
        return formatted
    
    def date_range_to_utc_intervals(self, date_range, interval_days):
        start_date, end_date = self.parse_date_range(date_range)
        intervals = self.generate_intervals(start_date, end_date, interval_days)
        utc_intervals = [(self.format_date_utc(start, True), self.format_date_utc(end, False)) for start, end in intervals]
        return utc_intervals

    def split_date_range_utc(self, start_str, end_str, fraction):

        start = urllib.parse.unquote(start_str)
        start = datetime.fromisoformat(start.replace('Z', '+00:00'))

        end = urllib.parse.unquote(end_str)
        end = datetime.fromisoformat(end.replace('Z', '+00:00'))

        total_duration = (end - start).total_seconds()
        interval_duration = total_duration * fraction

        intervals = []
        current_start = start

        while current_start < end:
            current_end = current_start + timedelta(seconds=interval_duration)
            if current_end > end:
                current_end = end

            start_iso = current_start.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_iso = current_end.strftime('%Y-%m-%dT%H:%M:%SZ')

            intervals.append(
                (
                    urllib.parse.quote(start_iso),
                    urllib.parse.quote(end_iso)
                )
            )
            current_start = current_end

        return intervals
    
    def change_format(self, data):
        for i in range(0, len(data), 2):
            data.loc[i, 'Direction'] = 'Pull'
            data.loc[i+1, 'Direction'] = 'Squeeze'
            data.loc[i, 'L Reps'] = data.loc[i, 'outerLeftRepetitions']
            data.loc[i, 'R Reps'] = data.loc[i, 'outerRightRepetitions']
            data.loc[i+1, 'L Reps'] = data.loc[i, 'innerLeftRepetitions']
            data.loc[i+1, 'R Reps'] = data.loc[i, 'outerRightRepetitions']
            data.loc[i, 'L Max Force (N)'] = data.loc[i, 'outerLeftMaxForce']
            data.loc[i, 'R Max Force (N)'] = data.loc[i, 'outerRightMaxForce']
            data.loc[i+1, 'L Max Force (N)'] = data.loc[i, 'innerLeftMaxForce']
            data.loc[i+1, 'R Max Force (N)'] = data.loc[i, 'innerRightMaxForce']
            data.loc[i, 'Max Imbalance'] = (data.loc[i, 'L Max Force (N)'] - data.loc[i, 'R Max Force (N)']) / max(data.loc[i, 'L Max Force (N)'], data.loc[i, 'R Max Force (N)'])
            data.loc[i+1, 'Max Imbalance'] = (data.loc[i+1, 'L Max Force (N)'] - data.loc[i+1, 'R Max Force (N)']) / max(data.loc[i+1, 'L Max Force (N)'], data.loc[i+1, 'R Max Force (N)'])
            l_max_ratio = round(data.loc[i+1, 'L Max Force (N)'] / data.loc[i, 'L Max Force (N)'], 2)
            r_max_ratio = round(data.loc[i+1, 'R Max Force (N)'] / data.loc[i, 'R Max Force (N)'], 2)
            data.loc[i, 'L Max Ratio'] = l_max_ratio
            data.loc[i, 'R Max Ratio'] = r_max_ratio
            data.loc[i+1, 'L Max Ratio'] = l_max_ratio
            data.loc[i+1, 'R Max Ratio'] = r_max_ratio
            data.loc[i, 'L Avg Force (N)'] = data.loc[i, 'outerLeftAvgForce']
            data.loc[i, 'R Avg Force (N)'] = data.loc[i, 'outerRightAvgForce']
            data.loc[i+1, 'L Avg Force (N)'] = data.loc[i+1, 'outerLeftAvgForce']
            data.loc[i+1, 'R Avg Force (N)'] = data.loc[i+1, 'outerRightAvgForce']
            data.loc[i, 'Avg Imbalance'] = (data.loc[i, 'L Avg Force (N)'] - data.loc[i, 'R Avg Force (N)']) / max(data.loc[i, 'L Avg Force (N)'], data.loc[i, 'R Avg Force (N)'])
            data.loc[i+1, 'Avg Imbalance'] =  (data.loc[i+1, 'L Avg Force (N)'] - data.loc[i+1, 'R Avg Force (N)']) / max(data.loc[i+1, 'L Avg Force (N)'], data.loc[i+1, 'R Avg Force (N)'])
            l_avg_ratio = round(data.loc[i+1, 'L Avg Force (N)'] / data.loc[i, 'L Avg Force (N)'], 2)
            r_avg_ratio = round(data.loc[i+1, 'R Avg Force (N)'] / data.loc[i, 'R Avg Force (N)'], 2)
            data.loc[i, 'L Avg Ratio'] = l_avg_ratio
            data.loc[i, 'R Avg Ratio'] = r_avg_ratio
            data.loc[i+1, 'L Avg Ratio'] = l_avg_ratio
            data.loc[i+1, 'R Avg Ratio'] = r_avg_ratio
            data.loc[i, 'L Max Impulse (Ns)'] = data.loc[i, 'outerLeftImpulse']
            data.loc[i, 'R Max Impulse (Ns)'] = data.loc[i, 'outerRightImpulse']
            data.loc[i+1, 'L Max Impulse (Ns)'] = data.loc[i, 'innerLeftImpulse']
            data.loc[i+1, 'R Max Impulse (Ns)'] = data.loc[i, 'innerRightImpulse']
            data.loc[i, 'Impulse Imbalance (%)'] = (data.loc[i, 'L Max Impulse (Ns)'] - data.loc[i, 'R Max Impulse (Ns)']) / max(data.loc[i, 'L Max Impulse (Ns)'], data.loc[i, 'R Max Impulse (Ns)'])
            data.loc[i+1, 'Impulse Imbalance (%)'] = (data.loc[i+1, 'L Max Impulse (Ns)'] - data.loc[i+1, 'R Max Impulse (Ns)']) / max(data.loc[i+1, 'L Max Impulse (Ns)'], data.loc[i+1, 'R Max Impulse (Ns)'])
        return data
    
    def fetch_data_recursively(self, start_date, end_date, granularity=1):
        intervals = self.split_date_range_utc(start_date, end_date, granularity)
        all_data = pd.DataFrame()
        print(intervals, "intervals")
        
        for start, end in intervals:
            data = self.get_tests((start.replace("%3A", ":"), end.replace("%3A", ":")))
            print(data, len(data), "appapap")
            print(start.replace("%3A", ":"), end.replace("%3A", ":"), len(data))
            if len(data) >= 50:
                smaller_data = self.fetch_data_recursively(start, end, granularity / 2)
                all_data = pd.concat([all_data, smaller_data])
            else:
                all_data = pd.concat([all_data, data])
        
        return all_data

    def get_data(self, date_range):
        start_date, end_date = date_range[0], date_range[1]
        df = self.fetch_data_recursively(start_date, end_date)
        if len(df) == 0:
            return None
        df = self.modify_df(df)
        df = self.change_format(df)
        return df
    
    def update_forceframe(self):

        last_update, last_index = self.get_last_update(self.vald_master_file_path)

        current_time = datetime.utcnow()
        future_time = current_time + timedelta(minutes=10)
        formatted_time = future_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        date_range = [last_update, formatted_time]
        print(date_range, "date_range")
        new_data = self.get_data(date_range)
        if new_data is None:
            return new_data
        else:
            new_data = new_data.sort_values(by='testDateUtc', ascending=True)
        
        new_start_index = last_index + 1
        new_indices = range(new_start_index, new_start_index + len(new_data))
        new_data.index = new_indices
        
        return new_data

    def update_master_file(self, new_data):
        
        if os.path.exists(self.vald_master_file_path):
            with open(self.vald_master_file_path, 'a') as f:
                f.write('\n')
            new_data.to_csv(self.vald_master_file_path, mode='a', header=False, index=True)
        else:
            new_data.to_csv(self.vald_master_file_path, index=True)
        
        print(f"Updated {self.vald_master_file_path}")

    def save_dataframes(self, teams_data):
        today_date = datetime.today().strftime('%Y-%m-%d')
        
        for team_name, test_data in teams_data.items():
            if not isinstance(team_name, str):
                team_name = str(team_name)
            
            sanitized_team_name = self.sanitize_foldername(team_name.lower())
            
            team_folder = os.path.join(self.base_directory, sanitized_team_name)
            if not os.path.exists(team_folder):
                os.makedirs(team_folder)
            
            for test_type, df in test_data.items():
                existing_file_path = None
                for file in os.listdir(team_folder):
                    if file.startswith(f"{self.sanitize_filename(team_name.lower()).replace('/', '-')}_{test_type.lower().replace(' ', '_').replace('/', '-')}_"):
                        existing_file_path = os.path.join(team_folder, file)
                        break
                
                if existing_file_path:
                    existing_df = pd.read_csv(existing_file_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                    os.remove(existing_file_path)
                
                raw_file_name = f"{team_name}_{test_type}_{today_date}".replace('/', '-').lower()
                sanitized_file_name = self.sanitize_filename(raw_file_name) + '.csv'
                new_file_path = os.path.join(team_folder, sanitized_file_name)
                
                df.to_csv(new_file_path, index=False)
                print(f"Saved {new_file_path}")

    def save_master_file(self, data):
        directory = os.path.dirname(self.vald_master_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data.to_csv(self.vald_master_file_path, index=True)

    def initial_setup(self):
        current_datetime = datetime.utcnow()

        if current_datetime < datetime(current_datetime.year, 8, 1): # start of athletic year - August 1
            year_for_august = current_datetime.year - 1
        else:
            year_for_august = current_datetime.year

        august_first_datetime = datetime(year_for_august, 8, 1)

        diff_in_months = (current_datetime.year - august_first_datetime.year) * 12 + current_datetime.month - august_first_datetime.month

        if abs(diff_in_months) > 6: # vald api cannot accept requests more than 6 months apart

            split_date = august_first_datetime + timedelta(days=180)
            split_date_formatted = split_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            august_first_formatted = august_first_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            one_hour_later_formatted = (current_datetime + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            formatted_dates_first = [august_first_formatted, split_date_formatted]
            formatted_dates_second = [split_date + timedelta(seconds=1), current_datetime]

            formatted_dates_second = [date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' if isinstance(date, datetime) else date for date in formatted_dates_second]

            print(formatted_dates_first, formatted_dates_second, "formatted_dates")

            data_first = self.get_data(formatted_dates_first)
            data_second = self.get_data(formatted_dates_second)
            
            data = pd.concat([data_first, data_second], ignore_index=True)
        else:
            august_first_formatted = august_first_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            one_hour_later_formatted = (current_datetime + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            formatted_dates = [august_first_formatted, one_hour_later_formatted]

            data = self.get_data(formatted_dates)

        self.save_master_file(data)

        teams_data = self.data_to_groups(data)

        self.save_dataframes(teams_data)

    def data_to_groups(self, data):
        teams_data = {}

        for group in data['Groups'].unique():
            teams_data[group] = {}
            group_data = data[data['Groups'] == group]
            
            for test in group_data['Test'].unique():
                test_data = group_data[group_data['Test'] == test].reset_index(drop=True)
                teams_data[group][test] = test_data

        return teams_data

    def populate_folders(self):
        if os.path.exists(self.vald_master_file_path) == False:
            print("Setting up intial")
            self.initial_setup()
        new_data = self.update_forceframe()
        if new_data is None:
            return None
        self.update_master_file(new_data)
        teams_data = self.data_to_groups(new_data)

        self.save_dataframes(teams_data)