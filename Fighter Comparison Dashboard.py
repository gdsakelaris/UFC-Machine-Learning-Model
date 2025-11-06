# UFC Fighter Comparison GUI Dashboard
"""
UFC Fighter Comparison GUI Dashboard
===================================

This script provides a GUI interface for comparing two UFC fighters side-by-side with 
comprehensive career statistics including striking distribution, grappling metrics, and fight history.

Features:
- Searchable fighter selection with autocomplete dropdown
- Side-by-side fighter comparison
- Comprehensive statistics display
- Complete fight history

Usage:
    python Fighter_Comparison_GUI.py

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For table formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not installed. Install with: pip install tabulate")

class FighterComparisonGUI:
    def __init__(self, root, data_file='Models/fight_data.csv'):
        self.root = root
        self.root.title("UFC Fighter Comparison Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Load fight data
        print("Loading fight data...")
        try:
            self.df = pd.read_csv(data_file)
            self.df['event_date'] = pd.to_datetime(self.df['event_date'])
            self.df = self.df.sort_values('event_date')
            print(f"Loaded {len(self.df)} fights from {self.df['event_date'].min().strftime('%Y-%m-%d')} to {self.df['event_date'].max().strftime('%Y-%m-%d')}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Could not find '{data_file}' file.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")
            return
        
        # Get unique fighter names
        self.all_fighters = sorted(set(self.df['r_fighter'].tolist() + self.df['b_fighter'].tolist()))
        
        # Initialize fighter data
        self.fighter1_data = None
        self.fighter2_data = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="UFC FIGHTER COMPARISON DASHBOARD", 
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Search frame
        search_frame = tk.Frame(self.root, bg='#2c3e50')
        search_frame.pack(pady=20)
        
        # Fighter 1 search
        tk.Label(search_frame, text="Fighter 1:", font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white').grid(row=0, column=0, padx=10, pady=5)
        self.fighter1_var = tk.StringVar()
        self.fighter1_combo = ttk.Combobox(
            search_frame, 
            textvariable=self.fighter1_var,
            font=('Arial', 11),
            width=30
        )
        self.fighter1_combo.grid(row=0, column=1, padx=10, pady=5)
        self.fighter1_combo['values'] = self.all_fighters
        self.fighter1_combo.bind('<KeyRelease>', lambda e: self.filter_fighters(1))
        self.fighter1_combo.bind('<<ComboboxSelected>>', lambda e: self.load_fighter(1))
        
        # Fighter 2 search
        tk.Label(search_frame, text="Fighter 2:", font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white').grid(row=0, column=2, padx=10, pady=5)
        self.fighter2_var = tk.StringVar()
        self.fighter2_combo = ttk.Combobox(
            search_frame, 
            textvariable=self.fighter2_var,
            font=('Arial', 11),
            width=30
        )
        self.fighter2_combo.grid(row=0, column=3, padx=10, pady=5)
        self.fighter2_combo['values'] = self.all_fighters
        self.fighter2_combo.bind('<KeyRelease>', lambda e: self.filter_fighters(2))
        self.fighter2_combo.bind('<<ComboboxSelected>>', lambda e: self.load_fighter(2))
        
        # Compare button
        compare_btn = tk.Button(
            search_frame,
            text="Compare Fighters",
            command=self.compare_fighters,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=5
        )
        compare_btn.grid(row=0, column=4, padx=10, pady=5)
        
        # Results frame with scrollbar
        results_frame = tk.Frame(self.root, bg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create scrollable text widget
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 10),
            bg='#34495e',
            fg='white',
            wrap=tk.WORD,
            width=120,
            height=40
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select two fighters to compare")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7',
            anchor='w'
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def filter_fighters(self, fighter_num):
        """Filter fighters based on search input"""
        search_term = self.fighter1_var.get() if fighter_num == 1 else self.fighter2_var.get()
        
        if len(search_term) < 2:
            combo = self.fighter1_combo if fighter_num == 1 else self.fighter2_combo
            combo['values'] = []
            return
            
        # Filter fighters that contain the search term
        filtered_fighters = [fighter for fighter in self.all_fighters 
                           if search_term.lower() in fighter.lower()]
        
        combo = self.fighter1_combo if fighter_num == 1 else self.fighter2_combo
        combo['values'] = filtered_fighters[:20]  # Limit to 20 results
        
        # Don't automatically open dropdown - let user control it
    
    def load_fighter(self, fighter_num):
        """Load fighter data when selected"""
        fighter_name = self.fighter1_var.get() if fighter_num == 1 else self.fighter2_var.get()
        
        if not fighter_name:
            return
            
        try:
            fighter_data = self.get_fighter_data(fighter_name)
            if fighter_data:
                if fighter_num == 1:
                    self.fighter1_data = fighter_data
                    self.status_var.set(f"Loaded: {fighter_name}")
                else:
                    self.fighter2_data = fighter_data
                    self.status_var.set(f"Loaded: {fighter_name}")
            else:
                messagebox.showwarning("Warning", f"Fighter '{fighter_name}' not found in database.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading fighter data: {e}")
    
    def compare_fighters(self):
        """Compare the two selected fighters"""
        if not self.fighter1_data or not self.fighter2_data:
            messagebox.showwarning("Warning", "Please select both fighters before comparing.")
            return
            
        try:
            self.status_var.set("Generating comparison...")
            self.root.update()
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            
            # Generate comparison
            comparison_text = self.generate_comparison_text(self.fighter1_data, self.fighter2_data)
            
            # Display results
            self.results_text.insert(tk.END, comparison_text)
            self.status_var.set("Comparison complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating comparison: {e}")
            self.status_var.set("Error occurred")
    
    def get_fighter_data(self, fighter_name):
        """Get all fights for a specific fighter and calculate career statistics"""
        # Find all fights where the fighter was either red or blue corner
        red_fights = self.df[self.df['r_fighter'].str.contains(fighter_name, case=False, na=False)]
        blue_fights = self.df[self.df['b_fighter'].str.contains(fighter_name, case=False, na=False)]
        
        if red_fights.empty and blue_fights.empty:
            return None
            
        # Combine and sort by date
        fighter_fights = []
        
        for _, fight in red_fights.iterrows():
            fight_data = {
                'date': fight['event_date'],
                'opponent': fight['b_fighter'],
                'result': 'Win' if fight['winner'] == 'Red' else 'Loss' if fight['winner'] == 'Blue' else 'Draw',
                'method': fight['method'],
                'round': fight['finish_round'],
                'time': fight['time_sec'],
                'total_time': fight['total_fight_time_sec'],
                'is_red': True,
                'height': fight['r_height'],
                'reach': fight['r_reach'],
                'stance': fight['r_stance'],
                'weight': fight['r_weight'],
                'age_at_fight': fight['r_age_at_event'],
                'date_of_birth': fight['r_date_of_birth'],
                'wins_before': fight['r_wins'],
                'losses_before': fight['r_losses'],
                'draws_before': fight['r_draws'],
                # Striking stats
                'sig_str_landed': fight['r_sig_str'],
                'sig_str_attempted': fight['r_sig_str_att'],
                'sig_str_accuracy': fight['r_sig_str_acc'],
                'total_str_landed': fight['r_str'],
                'total_str_attempted': fight['r_str_att'],
                'total_str_accuracy': fight['r_str_acc'],
                'knockdowns': fight['r_kd'],
                # Strike distribution percentages
                'head_pct': fight['r_head'],
                'body_pct': fight['r_body'],
                'leg_pct': fight['r_leg'],
                'distance_pct': fight['r_distance'],
                'clinch_pct': fight['r_clinch'],
                'ground_pct': fight['r_ground'],
                # Grappling stats
                'takedowns_landed': fight['r_td'],
                'takedowns_attempted': fight['r_td_att'],
                'takedown_accuracy': fight['r_td_acc'],
                'submission_attempts': fight['r_sub_att'],
                'reversals': fight['r_rev'],
                'control_time': fight['r_ctrl_sec'],
                # Career averages going into this fight
                'career_slpm': fight['r_pro_SLpM'],
                'career_sig_str_acc': fight['r_pro_sig_str_acc'],
                'career_sapm': fight['r_pro_SApM'],
                'career_str_def': fight['r_pro_str_def'],
                'career_td_avg': fight['r_pro_td_avg'],
                'career_td_acc': fight['r_pro_td_acc'],
                'career_td_def': fight['r_pro_td_def'],
                'career_sub_avg': fight['r_pro_sub_avg']
            }
            fighter_fights.append(fight_data)
            
        for _, fight in blue_fights.iterrows():
            fight_data = {
                'date': fight['event_date'],
                'opponent': fight['r_fighter'],
                'result': 'Win' if fight['winner'] == 'Blue' else 'Loss' if fight['winner'] == 'Red' else 'Draw',
                'method': fight['method'],
                'round': fight['finish_round'],
                'time': fight['time_sec'],
                'total_time': fight['total_fight_time_sec'],
                'is_red': False,
                'height': fight['b_height'],
                'reach': fight['b_reach'],
                'stance': fight['b_stance'],
                'weight': fight['b_weight'],
                'age_at_fight': fight['b_age_at_event'],
                'date_of_birth': fight['b_date_of_birth'],
                'wins_before': fight['b_wins'],
                'losses_before': fight['b_losses'],
                'draws_before': fight['b_draws'],
                # Striking stats
                'sig_str_landed': fight['b_sig_str'],
                'sig_str_attempted': fight['b_sig_str_att'],
                'sig_str_accuracy': fight['b_sig_str_acc'],
                'total_str_landed': fight['b_str'],
                'total_str_attempted': fight['b_str_att'],
                'total_str_accuracy': fight['b_str_acc'],
                'knockdowns': fight['b_kd'],
                # Strike distribution percentages
                'head_pct': fight['b_head'],
                'body_pct': fight['b_body'],
                'leg_pct': fight['b_leg'],
                'distance_pct': fight['b_distance'],
                'clinch_pct': fight['b_clinch'],
                'ground_pct': fight['b_ground'],
                # Grappling stats
                'takedowns_landed': fight['b_td'],
                'takedowns_attempted': fight['b_td_att'],
                'takedown_accuracy': fight['b_td_acc'],
                'submission_attempts': fight['b_sub_att'],
                'reversals': fight['b_rev'],
                'control_time': fight['b_ctrl_sec'],
                # Career averages going into this fight
                'career_slpm': fight['b_pro_SLpM'],
                'career_sig_str_acc': fight['b_pro_sig_str_acc'],
                'career_sapm': fight['b_pro_SApM'],
                'career_str_def': fight['b_pro_str_def'],
                'career_td_avg': fight['b_pro_td_avg'],
                'career_td_acc': fight['b_pro_td_acc'],
                'career_td_def': fight['b_pro_td_def'],
                'career_sub_avg': fight['b_pro_sub_avg']
            }
            fighter_fights.append(fight_data)
        
        # Convert to DataFrame and sort by date
        fighter_df = pd.DataFrame(fighter_fights)
        fighter_df = fighter_df.sort_values('date').reset_index(drop=True)
        
        if fighter_df.empty:
            return None
            
        return self.calculate_career_stats(fighter_df, fighter_name)
    
    def calculate_career_stats(self, fighter_df, fighter_name):
        """Calculate comprehensive career statistics for a fighter"""
        stats = {
            'name': fighter_name,
            'total_fights': len(fighter_df),
            'wins': len(fighter_df[fighter_df['result'] == 'Win']),
            'losses': len(fighter_df[fighter_df['result'] == 'Loss']),
            'draws': len(fighter_df[fighter_df['result'] == 'Draw']),
            'fight_history': []
        }
        
        # Basic info from most recent fight
        latest_fight = fighter_df.iloc[-1]
        stats['height'] = latest_fight['height']
        stats['reach'] = latest_fight['reach']
        stats['stance'] = latest_fight['stance']
        stats['weight'] = latest_fight['weight']
        
        # Calculate current age
        if pd.notna(latest_fight['date_of_birth']):
            try:
                dob = pd.to_datetime(latest_fight['date_of_birth'])
                stats['current_age'] = (datetime.now() - dob).days // 365
            except (ValueError, TypeError):
                stats['current_age'] = latest_fight['age_at_fight'] + (datetime.now().year - latest_fight['date'].year)
        else:
            stats['current_age'] = latest_fight['age_at_fight'] + (datetime.now().year - latest_fight['date'].year)
        
        # Fight history
        for _, fight in fighter_df.iterrows():
            fight_info = {
                'date': fight['date'].strftime('%Y-%m-%d'),
                'opponent': fight['opponent'],
                'result': fight['result'],
                'method': fight['method'],
                'round': fight['round'],
                'time': fight['time']
            }
            stats['fight_history'].append(fight_info)
        
        # Calculate method breakdown for wins and losses
        wins_df = fighter_df[fighter_df['result'] == 'Win']
        losses_df = fighter_df[fighter_df['result'] == 'Loss']
        
        stats['wins_by_method'] = wins_df['method'].value_counts().to_dict()
        stats['losses_by_method'] = losses_df['method'].value_counts().to_dict()
        
        # Fill in missing methods with 0
        all_methods = ['KO/TKO', 'Submission', 'Decision', 'DQ', 'No Contest']
        for method in all_methods:
            if method not in stats['wins_by_method']:
                stats['wins_by_method'][method] = 0
            if method not in stats['losses_by_method']:
                stats['losses_by_method'][method] = 0
        
        # Calculate career striking totals by distribution
        total_sig_str_landed = fighter_df['sig_str_landed'].sum()
        
        # Calculate strike distribution totals
        stats['strike_distribution_totals'] = {
            'head_landed': (fighter_df['sig_str_landed'] * fighter_df['head_pct']).sum(),
            'body_landed': (fighter_df['sig_str_landed'] * fighter_df['body_pct']).sum(),
            'leg_landed': (fighter_df['sig_str_landed'] * fighter_df['leg_pct']).sum(),
            'distance_landed': (fighter_df['sig_str_landed'] * fighter_df['distance_pct']).sum(),
            'clinch_landed': (fighter_df['sig_str_landed'] * fighter_df['clinch_pct']).sum(),
            'ground_landed': (fighter_df['sig_str_landed'] * fighter_df['ground_pct']).sum()
        }
        
        # Calculate absorbed strikes by looking at opponent data
        total_sig_str_absorbed = 0
        stats['strike_distribution_absorbed'] = {
            'head_absorbed': 0,
            'body_absorbed': 0,
            'leg_absorbed': 0,
            'distance_absorbed': 0,
            'clinch_absorbed': 0,
            'ground_absorbed': 0
        }
        
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    opp_sig_str = opponent_fight.iloc[0]['b_sig_str']
                    total_sig_str_absorbed += opp_sig_str
                    # Calculate absorbed strike distribution
                    stats['strike_distribution_absorbed']['head_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_head']
                    stats['strike_distribution_absorbed']['body_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_body']
                    stats['strike_distribution_absorbed']['leg_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_leg']
                    stats['strike_distribution_absorbed']['distance_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_distance']
                    stats['strike_distribution_absorbed']['clinch_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_clinch']
                    stats['strike_distribution_absorbed']['ground_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_ground']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    opp_sig_str = opponent_fight.iloc[0]['r_sig_str']
                    total_sig_str_absorbed += opp_sig_str
                    # Calculate absorbed strike distribution
                    stats['strike_distribution_absorbed']['head_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_head']
                    stats['strike_distribution_absorbed']['body_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_body']
                    stats['strike_distribution_absorbed']['leg_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_leg']
                    stats['strike_distribution_absorbed']['distance_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_distance']
                    stats['strike_distribution_absorbed']['clinch_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_clinch']
                    stats['strike_distribution_absorbed']['ground_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_ground']
        
        # Calculate total significant strikes landed
        stats['total_sig_str_landed'] = total_sig_str_landed
        stats['total_sig_str_absorbed'] = total_sig_str_absorbed
        stats['avg_sig_str_landed_per_fight'] = total_sig_str_landed / stats['total_fights'] if stats['total_fights'] > 0 else 0
        stats['avg_sig_str_absorbed_per_fight'] = total_sig_str_absorbed / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate averages per fight
        stats['strike_distribution_avg_per_fight'] = {
            'head_avg': stats['strike_distribution_totals']['head_landed'] / stats['total_fights'],
            'body_avg': stats['strike_distribution_totals']['body_landed'] / stats['total_fights'],
            'leg_avg': stats['strike_distribution_totals']['leg_landed'] / stats['total_fights'],
            'distance_avg': stats['strike_distribution_totals']['distance_landed'] / stats['total_fights'],
            'clinch_avg': stats['strike_distribution_totals']['clinch_landed'] / stats['total_fights'],
            'ground_avg': stats['strike_distribution_totals']['ground_landed'] / stats['total_fights']
        }
        
        # Calculate absorbed strike averages per fight
        stats['strike_distribution_absorbed_avg_per_fight'] = {
            'head_absorbed_avg': stats['strike_distribution_absorbed']['head_absorbed'] / stats['total_fights'],
            'body_absorbed_avg': stats['strike_distribution_absorbed']['body_absorbed'] / stats['total_fights'],
            'leg_absorbed_avg': stats['strike_distribution_absorbed']['leg_absorbed'] / stats['total_fights'],
            'distance_absorbed_avg': stats['strike_distribution_absorbed']['distance_absorbed'] / stats['total_fights'],
            'clinch_absorbed_avg': stats['strike_distribution_absorbed']['clinch_absorbed'] / stats['total_fights'],
            'ground_absorbed_avg': stats['strike_distribution_absorbed']['ground_absorbed'] / stats['total_fights']
        }
        
        # Career averages (from the data)
        latest_career_avgs = fighter_df.iloc[-1]
        stats['career_averages'] = {
            'slpm': latest_career_avgs['career_slpm'],
            'sig_str_accuracy': latest_career_avgs['career_sig_str_acc'],
            'sapm': latest_career_avgs['career_sapm'],
            'str_defense': latest_career_avgs['career_str_def'],
            'td_avg_per_15min': latest_career_avgs['career_td_avg'],
            'td_accuracy': latest_career_avgs['career_td_acc'],
            'td_defense': latest_career_avgs['career_td_def'],
            'sub_avg_per_15min': latest_career_avgs['career_sub_avg']
        }
        
        # Calculate fight time statistics
        total_fight_time = fighter_df['total_time'].sum()
        stats['avg_fight_time'] = total_fight_time / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Control time statistics
        total_control_time = fighter_df['control_time'].sum()
        stats['total_control_time'] = total_control_time
        stats['avg_control_time_per_fight'] = total_control_time / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Takedown statistics
        total_takedowns_landed = fighter_df['takedowns_landed'].sum()
        total_takedowns_attempted = fighter_df['takedowns_attempted'].sum()
        stats['total_takedowns_landed'] = total_takedowns_landed
        stats['total_takedowns_attempted'] = total_takedowns_attempted
        stats['career_takedown_accuracy'] = (total_takedowns_landed / total_takedowns_attempted) if total_takedowns_attempted > 0 else 0
        stats['avg_takedowns_per_fight'] = total_takedowns_landed / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate takedowns against by looking at opponent data
        total_takedowns_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_takedowns_against += opponent_fight.iloc[0]['b_td']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_takedowns_against += opponent_fight.iloc[0]['r_td']
        
        stats['total_takedowns_against'] = total_takedowns_against
        stats['avg_takedowns_against_per_fight'] = total_takedowns_against / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate knockdowns for and against
        total_knockdowns_landed = fighter_df['knockdowns'].sum()
        stats['total_knockdowns_landed'] = total_knockdowns_landed
        stats['avg_knockdowns_per_fight'] = total_knockdowns_landed / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate knockdowns against by looking at opponent data
        total_knockdowns_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_knockdowns_against += opponent_fight.iloc[0]['b_kd']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_knockdowns_against += opponent_fight.iloc[0]['r_kd']
        
        stats['total_knockdowns_against'] = total_knockdowns_against
        stats['avg_knockdowns_against_per_fight'] = total_knockdowns_against / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate submission attempts for and against
        total_submission_attempts = fighter_df['submission_attempts'].sum()
        stats['total_submission_attempts'] = total_submission_attempts
        stats['avg_submission_attempts_per_fight'] = total_submission_attempts / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate submission attempts against by looking at opponent data
        total_submission_attempts_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_submission_attempts_against += opponent_fight.iloc[0]['b_sub_att']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_submission_attempts_against += opponent_fight.iloc[0]['r_sub_att']
        
        stats['total_submission_attempts_against'] = total_submission_attempts_against
        stats['avg_submission_attempts_against_per_fight'] = total_submission_attempts_against / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate reversals for and against
        total_reversals = fighter_df['reversals'].sum()
        stats['total_reversals'] = total_reversals
        stats['avg_reversals_per_fight'] = total_reversals / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate reversals against by looking at opponent data
        total_reversals_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_reversals_against += opponent_fight.iloc[0]['b_rev']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_reversals_against += opponent_fight.iloc[0]['r_rev']
        
        stats['total_reversals_against'] = total_reversals_against
        stats['avg_reversals_against_per_fight'] = total_reversals_against / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        # Calculate control time against by looking at opponent data
        total_control_time_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                # Fighter was red corner, opponent was blue corner
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) & 
                    (self.df['b_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_control_time_against += opponent_fight.iloc[0]['b_ctrl_sec']
            else:
                # Fighter was blue corner, opponent was red corner
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) & 
                    (self.df['r_fighter'] == fight['opponent']) & 
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_control_time_against += opponent_fight.iloc[0]['r_ctrl_sec']
        
        stats['total_control_time_against'] = total_control_time_against
        stats['avg_control_time_against_per_fight'] = total_control_time_against / stats['total_fights'] if stats['total_fights'] > 0 else 0
        
        return stats
    
    def format_table(self, data, headers):
        """Format data as a table, using tabulate if available, otherwise basic formatting"""
        if HAS_TABULATE:
            return tabulate(data, headers=headers, tablefmt='grid', stralign='left')
        else:
            # Basic fallback formatting
            result = ""
            # Calculate column widths
            col_widths = [len(header) for header in headers]
            for row in data:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Create header
            header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
            result += header_line + "\n"
            result += "-" * len(header_line) + "\n"
            
            # Create data rows
            for row in data:
                row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                result += row_line + "\n"
            
            return result
    
    def generate_comparison_text(self, fighter1_stats, fighter2_stats):
        """Generate the comparison text for display"""
        if not fighter1_stats or not fighter2_stats:
            return "Error: Could not find data for one or both fighters."
        
        text = ""
        text += "="*120 + "\n"
        text += f"{'UFC FIGHTER COMPARISON DASHBOARD':^120}\n"
        text += "="*120 + "\n\n"
        
        # Check if tabulate is available
        if not HAS_TABULATE:
            text += "WARNING: tabulate package not installed. Install with: pip install tabulate\n"
            text += "Displaying in basic format without tables.\n\n"
        
        # Basic Information
        text += f"{'BASIC INFORMATION':<120}\n"
        basic_data = [
            ['Name', fighter1_stats['name'], fighter2_stats['name']],
            ['Height (inches)', fighter1_stats['height'], fighter2_stats['height']],
            ['Reach (inches)', fighter1_stats['reach'], fighter2_stats['reach']],
            ['Stance', fighter1_stats['stance'], fighter2_stats['stance']],
            ['Current Age', fighter1_stats['current_age'], fighter2_stats['current_age']],
            ['Weight (lbs)', fighter1_stats['weight'], fighter2_stats['weight']]
        ]
        text += self.format_table(basic_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Fight Record
        text += f"{'FIGHT RECORD':<120}\n"
        win_rate1 = (fighter1_stats['wins'] / fighter1_stats['total_fights'] * 100) if fighter1_stats['total_fights'] > 0 else 0
        win_rate2 = (fighter2_stats['wins'] / fighter2_stats['total_fights'] * 100) if fighter2_stats['total_fights'] > 0 else 0
        
        record_data = [
            ['Total Fights', fighter1_stats['total_fights'], fighter2_stats['total_fights']],
            ['Wins', fighter1_stats['wins'], fighter2_stats['wins']],
            ['Losses', fighter1_stats['losses'], fighter2_stats['losses']],
            ['Draws', fighter1_stats['draws'], fighter2_stats['draws']],
            ['Win Rate (%)', f"{win_rate1:.1f}%", f"{win_rate2:.1f}%"]
        ]
        text += self.format_table(record_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Wins and Losses by Method (combined)
        text += f"{'WINS & LOSSES BY METHOD':<120}\n"
        method_data = [
            ['KO/TKO', f"{fighter1_stats['wins_by_method']['KO/TKO']}-{fighter1_stats['losses_by_method']['KO/TKO']}", f"{fighter2_stats['wins_by_method']['KO/TKO']}-{fighter2_stats['losses_by_method']['KO/TKO']}"],
            ['Submission', f"{fighter1_stats['wins_by_method']['Submission']}-{fighter1_stats['losses_by_method']['Submission']}", f"{fighter2_stats['wins_by_method']['Submission']}-{fighter2_stats['losses_by_method']['Submission']}"],
            ['Decision', f"{fighter1_stats['wins_by_method']['Decision']}-{fighter1_stats['losses_by_method']['Decision']}", f"{fighter2_stats['wins_by_method']['Decision']}-{fighter2_stats['losses_by_method']['Decision']}"],
            ['DQ', f"{fighter1_stats['wins_by_method']['DQ']}-{fighter1_stats['losses_by_method']['DQ']}", f"{fighter2_stats['wins_by_method']['DQ']}-{fighter2_stats['losses_by_method']['DQ']}"],
            ['No Contest', f"{fighter1_stats['wins_by_method']['No Contest']}-{fighter1_stats['losses_by_method']['No Contest']}", f"{fighter2_stats['wins_by_method']['No Contest']}-{fighter2_stats['losses_by_method']['No Contest']}"]
        ]
        text += self.format_table(method_data, ['Method', f"{fighter1_stats['name']} (W-L)", f"{fighter2_stats['name']} (W-L)"])
        text += "\n\n"
        
        # Career Averages
        text += f"{'CAREER AVERAGES':<120}\n"
        career_data = [
            ['SLpM (Sig Strikes/Min)', f"{fighter1_stats['career_averages']['slpm']:.2f}", f"{fighter2_stats['career_averages']['slpm']:.2f}"],
            ['Sig Strike Accuracy (%)', f"{fighter1_stats['career_averages']['sig_str_accuracy']*100:.1f}%", f"{fighter2_stats['career_averages']['sig_str_accuracy']*100:.1f}%"],
            ['SApM (Sig Absorbed/Min)', f"{fighter1_stats['career_averages']['sapm']:.2f}", f"{fighter2_stats['career_averages']['sapm']:.2f}"],
            ['Strike Defense (%)', f"{fighter1_stats['career_averages']['str_defense']*100:.1f}%", f"{fighter2_stats['career_averages']['str_defense']*100:.1f}%"],
            ['TD per 15min', f"{fighter1_stats['career_averages']['td_avg_per_15min']:.2f}", f"{fighter2_stats['career_averages']['td_avg_per_15min']:.2f}"],
            ['TD Accuracy (%)', f"{fighter1_stats['career_averages']['td_accuracy']*100:.1f}%", f"{fighter2_stats['career_averages']['td_accuracy']*100:.1f}%"],
            ['TD Defense (%)', f"{fighter1_stats['career_averages']['td_defense']*100:.1f}%", f"{fighter2_stats['career_averages']['td_defense']*100:.1f}%"],
            ['Sub per 15min', f"{fighter1_stats['career_averages']['sub_avg_per_15min']:.2f}", f"{fighter2_stats['career_averages']['sub_avg_per_15min']:.2f}"]
        ]
        text += self.format_table(career_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Significant Strikes (Totals & Averages)
        text += f"{'SIGNIFICANT STRIKES':<120}\n"
        sig_strikes_data = [
            ['Total Landed', f"{fighter1_stats['total_sig_str_landed']:.0f}", f"{fighter2_stats['total_sig_str_landed']:.0f}"],
            ['Total Absorbed', f"{fighter1_stats['total_sig_str_absorbed']:.0f}", f"{fighter2_stats['total_sig_str_absorbed']:.0f}"],
            ['Avg Landed/Fight', f"{fighter1_stats['avg_sig_str_landed_per_fight']:.1f}", f"{fighter2_stats['avg_sig_str_landed_per_fight']:.1f}"],
            ['Avg Absorbed/Fight', f"{fighter1_stats['avg_sig_str_absorbed_per_fight']:.1f}", f"{fighter2_stats['avg_sig_str_absorbed_per_fight']:.1f}"]
        ]
        text += self.format_table(sig_strikes_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Strike Distribution (Landed - Totals & Averages)
        text += f"{'STRIKE DISTRIBUTION (Landed)':<120}\n"
        strike_landed_data = [
            ['Head (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['head_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['head_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['head_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['head_avg']:.1f}"],
            ['Body (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['body_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['body_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['body_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['body_avg']:.1f}"],
            ['Leg (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['leg_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['leg_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['leg_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['leg_avg']:.1f}"],
            ['Distance (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['distance_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['distance_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['distance_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['distance_avg']:.1f}"],
            ['Clinch (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['clinch_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['clinch_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['clinch_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['clinch_avg']:.1f}"],
            ['Ground (Total/Avg)', f"{fighter1_stats['strike_distribution_totals']['ground_landed']:.0f} / {fighter1_stats['strike_distribution_avg_per_fight']['ground_avg']:.1f}", f"{fighter2_stats['strike_distribution_totals']['ground_landed']:.0f} / {fighter2_stats['strike_distribution_avg_per_fight']['ground_avg']:.1f}"]
        ]
        text += self.format_table(strike_landed_data, ['Target/Position', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Strike Distribution (Absorbed - Totals & Averages)
        text += f"{'STRIKE DISTRIBUTION (Absorbed)':<120}\n"
        strike_absorbed_data = [
            ['Head (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['head_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['head_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['head_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['head_absorbed_avg']:.1f}"],
            ['Body (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['body_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['body_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['body_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['body_absorbed_avg']:.1f}"],
            ['Leg (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['leg_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['leg_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['leg_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['leg_absorbed_avg']:.1f}"],
            ['Distance (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['distance_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['distance_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['distance_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['distance_absorbed_avg']:.1f}"],
            ['Clinch (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['clinch_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['clinch_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['clinch_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['clinch_absorbed_avg']:.1f}"],
            ['Ground (Total/Avg)', f"{fighter1_stats['strike_distribution_absorbed']['ground_absorbed']:.0f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['ground_absorbed_avg']:.1f}", f"{fighter2_stats['strike_distribution_absorbed']['ground_absorbed']:.0f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['ground_absorbed_avg']:.1f}"]
        ]
        text += self.format_table(strike_absorbed_data, ['Target/Position', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Takedown Statistics (For & Against)
        text += f"{'TAKEDOWN STATISTICS':<120}\n"
        takedown_data = [
            ['Landed (Total/Avg)', f"{fighter1_stats['total_takedowns_landed']:.0f} / {fighter1_stats['avg_takedowns_per_fight']:.1f}", f"{fighter2_stats['total_takedowns_landed']:.0f} / {fighter2_stats['avg_takedowns_per_fight']:.1f}"],
            ['Attempted', f"{fighter1_stats['total_takedowns_attempted']:.0f}", f"{fighter2_stats['total_takedowns_attempted']:.0f}"],
            ['Accuracy (%)', f"{fighter1_stats['career_takedown_accuracy']*100:.1f}%", f"{fighter2_stats['career_takedown_accuracy']*100:.1f}%"],
            ['Against (Total/Avg)', f"{fighter1_stats['total_takedowns_against']:.0f} / {fighter1_stats['avg_takedowns_against_per_fight']:.1f}", f"{fighter2_stats['total_takedowns_against']:.0f} / {fighter2_stats['avg_takedowns_against_per_fight']:.1f}"]
        ]
        text += self.format_table(takedown_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Knockdown Statistics (For & Against)
        text += f"{'KNOCKDOWN STATISTICS':<120}\n"
        knockdown_data = [
            ['Landed (Total/Avg)', f"{fighter1_stats['total_knockdowns_landed']:.0f} / {fighter1_stats['avg_knockdowns_per_fight']:.1f}", f"{fighter2_stats['total_knockdowns_landed']:.0f} / {fighter2_stats['avg_knockdowns_per_fight']:.1f}"],
            ['Against (Total/Avg)', f"{fighter1_stats['total_knockdowns_against']:.0f} / {fighter1_stats['avg_knockdowns_against_per_fight']:.1f}", f"{fighter2_stats['total_knockdowns_against']:.0f} / {fighter2_stats['avg_knockdowns_against_per_fight']:.1f}"]
        ]
        text += self.format_table(knockdown_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Submission Attempt Statistics (For & Against)
        text += f"{'SUBMISSION ATTEMPT STATISTICS':<120}\n"
        submission_data = [
            ['Attempted (Total/Avg)', f"{fighter1_stats['total_submission_attempts']:.0f} / {fighter1_stats['avg_submission_attempts_per_fight']:.1f}", f"{fighter2_stats['total_submission_attempts']:.0f} / {fighter2_stats['avg_submission_attempts_per_fight']:.1f}"],
            ['Against (Total/Avg)', f"{fighter1_stats['total_submission_attempts_against']:.0f} / {fighter1_stats['avg_submission_attempts_against_per_fight']:.1f}", f"{fighter2_stats['total_submission_attempts_against']:.0f} / {fighter2_stats['avg_submission_attempts_against_per_fight']:.1f}"]
        ]
        text += self.format_table(submission_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Reversal Statistics (For & Against)
        text += f"{'REVERSAL STATISTICS':<120}\n"
        reversal_data = [
            ['Landed (Total/Avg)', f"{fighter1_stats['total_reversals']:.0f} / {fighter1_stats['avg_reversals_per_fight']:.1f}", f"{fighter2_stats['total_reversals']:.0f} / {fighter2_stats['avg_reversals_per_fight']:.1f}"],
            ['Against (Total/Avg)', f"{fighter1_stats['total_reversals_against']:.0f} / {fighter1_stats['avg_reversals_against_per_fight']:.1f}", f"{fighter2_stats['total_reversals_against']:.0f} / {fighter2_stats['avg_reversals_against_per_fight']:.1f}"]
        ]
        text += self.format_table(reversal_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Fight Time and Control (For & Against)
        text += f"{'FIGHT TIME & CONTROL':<120}\n"
        control_data = [
            ['Avg Fight Time (sec)', f"{fighter1_stats['avg_fight_time']:.0f}", f"{fighter2_stats['avg_fight_time']:.0f}"],
            ['Control Time (Total/Avg)', f"{fighter1_stats['total_control_time']:.0f} / {fighter1_stats['avg_control_time_per_fight']:.1f}", f"{fighter2_stats['total_control_time']:.0f} / {fighter2_stats['avg_control_time_per_fight']:.1f}"],
            ['Control Time Against (Total/Avg)', f"{fighter1_stats['total_control_time_against']:.0f} / {fighter1_stats['avg_control_time_against_per_fight']:.1f}", f"{fighter2_stats['total_control_time_against']:.0f} / {fighter2_stats['avg_control_time_against_per_fight']:.1f}"]
        ]
        text += self.format_table(control_data, ['Metric', fighter1_stats['name'], fighter2_stats['name']])
        text += "\n\n"
        
        # Fight History
        text += f"{'FIGHT HISTORY':<120}\n"
        text += "-"*120 + "\n"
        
        # Fighter 1 History
        text += f"\n{fighter1_stats['name']} - All Fights:\n"
        text += "-"*60 + "\n"
        text += f"{'Date':<12} {'Opponent':<25} {'Result':<8} {'Method':<15}\n"
        text += "-"*60 + "\n"
        for fight in reversed(fighter1_stats['fight_history']):
            text += f"{fight['date']:<12} {fight['opponent'][:24]:<25} {fight['result']:<8} {fight['method'][:14]:<15}\n"
        
        # Fighter 2 History
        text += f"\n{fighter2_stats['name']} - All Fights:\n"
        text += "-"*60 + "\n"
        text += f"{'Date':<12} {'Opponent':<25} {'Result':<8} {'Method':<15}\n"
        text += "-"*60 + "\n"
        for fight in reversed(fighter2_stats['fight_history']):
            text += f"{fight['date']:<12} {fight['opponent'][:24]:<25} {fight['result']:<8} {fight['method'][:14]:<15}\n"
        
        return text

def main():
    root = tk.Tk()
    FighterComparisonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
