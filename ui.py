"""
Program for interactive UI to provide ASP utility tools
"""
import streamlit as st
import pandas as pd
import re
import random

def generate_teams(num_teams):
    """Adapted from generator.py, generates a list of teams with a realistic confederation distribution."""
    CONFED_COUNTS = {'uefa': 13, 'afc': 6, 'caf': 5, 'conmebol': 4, 'concacaf': 3, 'ofc': 1}
    
    total_confed_slots = sum(CONFED_COUNTS.values())
    # Calculate proportional team allocation per confederation
    weights = {confed: int(round((count / total_confed_slots) * num_teams)) for confed, count in CONFED_COUNTS.items()}

    # Fix rounding issues to ensure exact num_teams
    while sum(weights.values()) < num_teams: weights[max(weights, key=lambda c: CONFED_COUNTS[c])] += 1
    while sum(weights.values()) > num_teams: 
        max_confed = max(weights, key=lambda c: weights[c])
        if weights[max_confed] > 0: weights[max_confed] -= 1

    # Build confed pool with adjusted weights
    confed_pool = sum([[c] * n for c, n in weights.items()], [])
    random.shuffle(confed_pool)

    teams = [(f"t{i+1}", confed_pool[i]) for i in range(num_teams)]
    team_facts = [f"team({t[0]})." for t in teams]
    confed_facts = [f"confed({t[0]}, {t[1]})." for t in teams]
    return teams, team_facts, confed_facts

# === PARSING LOGIC ===
def parse_clingo_result(content):
    """Parses raw Clingo solver output text into a dictionary of metrics."""
    data = {
        'solver': 'Clingo',
        'status': 'UNKNOWN',
        'time': None,
        'solving_time': None,
        'models': None,
        'optimum': 'unknown',
        'objective_value': None,
        'lower_bound': None,
        'upper_bound': None
    }
    
    # Pattern to find the main summary line for Time.
    time_match = re.search(
        r"Time\s*:\s*([\d.]+)s\s*\(Solving:\s*([\d.]+)s\s*1st Model:\s*([\d.]+)s\s*Unsat:\s*([\d.]+)s\)",
        content
    )
   
    # Extract Solutions found (Models), Optimum, and Optimization values
    summary_block_match = re.search(
        r"Models\s*:\s*(\d+)\+?\s*\n"
        r"\s*Optimum\s*:\s*(\w+)\s*\n"
        r"Optimization\s*:\s*(\d+)",
        content
    )
    
    bounds_match = re.search(r"Bounds\s*:\s*\[(\d+);(\d+)\]", content)
    
    if time_match:
        data['time'] = float(time_match.group(1))
        data['solving_time'] = float(time_match.group(2))
        data['status'] = 'COMPLETED'
    
    if summary_block_match:
        data['models'] = int(summary_block_match.group(1))
        data['optimum'] = summary_block_match.group(2)
        data['objective_value'] = int(summary_block_match.group(3))
    
    if bounds_match:
        data['lower_bound'] = int(bounds_match.group(1))
        data['upper_bound'] = int(bounds_match.group(2))
    
    return data

def parse_cpsat_result(content):
    """Parses raw CP-SAT solver output text into a dictionary of metrics"""
    data = {
        'solver': 'CP-SAT',
        'status': 'UNKNOWN',
        'time': None,
        'solving_time': None,
        'solutions_found': None,
        'conflicts': None,
        'branches': None,
        'objective_value': None
    }
    
    # Extract metrics from the final summary block
    summary_match = re.search(
        r"Status: (\w+)\n"
        r"Solutions found: (\d+)\n"
        r"Time: ([\d.]+) seconds\n"
        r"Conflicts: (\d+)\n"
        r"Branches: (\d+)\n"
        r"Best objective value found: ([\d.]+)",
        content
    )
    
    if summary_match:
        data['status'] = summary_match.group(1)
        data['solutions_found'] = int(summary_match.group(2))
        data['solving_time'] = float(summary_match.group(3))
        data['conflicts'] = int(summary_match.group(4))
        data['branches'] = int(summary_match.group(5))
        data['objective_value'] = float(summary_match.group(6))
    
    return data

# === UI Formatting Utilities ===
def map_group(group_code):
    """Convert group codes in facts to an alphabet"""
    if group_code.upper().startswith("G") and group_code[1:].isdigit():
        number = int(group_code[1:])
        return f"Group {chr(64 + number)}"  # 65 -> A, 66 -> B ...
    return group_code

def map_slot(slot):
    """Convert slot numbers into time strings"""
    mapping = {1: "10:00", 2: "13:00", 3: "16:00", 4: "19:00"}
    return mapping.get(slot, f"Slot {slot}")

def format_word(word):
    """Insert spaces before capital letters in a CamelCase string"""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', word)

# === UI Components ===

# Set page config
st.set_page_config(
    page_title="STASP",
    page_icon="🏆", 
    layout="centered"
)

st.title("Sports Timetabling with ASP")

# === TOURNAMENT INSTANCE GENERATOR ===
st.subheader("📝 Tournament Instance Generator")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Parameters")
    num_teams = st.number_input("Number of Teams", min_value=4, max_value=32, value=12, step=4)
    num_venues = st.number_input("Number of Venues", min_value=1, max_value=15, value=6, step=1)
    num_days = st.number_input("Number of Days", min_value=1, max_value=32, value=9, step=1)
    num_slots = st.number_input("Time Slots per Day", min_value=1, max_value=8, value=4, step=1)

with col2:
    st.markdown("##### Custom Facts (Override generated facts)")
    custom_teams = st.text_input("Custom Teams (comma-separated)", placeholder="brazil, germany, argentina, france",
        help="Leave empty to use default t1, t2, etc."
    )
    custom_venues = st.text_input("Custom Venues (comma-separated)", placeholder="wembley, maracana, allianz_arena",
        help="Leave empty to use default v1, v2, etc."
    )
    custom_host = st.text_input("Host Team", placeholder="brazil",
        help="Must match one of the team names"
    )
    additional_facts = st.text_area("Additional Facts", height=80,
        placeholder="venue_capacity(wembley, large).\nconfed(brazil, conmebol).",
        help="Any additional ASP facts"
    )

if st.button("Generate Instance File", type="primary"):
    if num_teams % 4 != 0:
        st.error("Number of teams must be divisible by 4")
    else:
        try:
            # Generate constants
            const_facts = [
                f"#const teams_per_group = 4.",
                f"#const num_groups = {num_teams // 4}.",
                f"#const num_teams = {num_teams}.",
                f"#const num_venues = {num_venues}."
            ]
            
            # Handle custom teams or generate default
            if custom_teams.strip():
                team_names = [name.strip() for name in custom_teams.split(',')]
                if len(team_names) != num_teams:
                    st.error(f"Please provide exactly {num_teams} team names")
                    st.stop()
                
                # Create team facts
                teams_data = [(name, None) for name in team_names]  # Will assign confederations later
                team_facts = [f"team({name})." for name in team_names]
                
                # Use generator's confederation logic
                teams_with_confed, _, confed_facts = generate_teams(num_teams)
                # Replace team names but keep confederation distribution
                final_confed_facts = []
                for i, (_, confed) in enumerate(teams_with_confed):
                    if i < len(team_names):
                        final_confed_facts.append(f"confed({team_names[i]},{confed}).")
                confed_facts = final_confed_facts
                
            else:
                # Use generator.py function
                teams_data, team_facts, confed_facts = generate_teams(num_teams)
                team_names = [t[0] for t in teams_data]
            
            # Generate rankings
            rank_facts = []
            for i, name in enumerate(team_names):
                rank_facts.append(f"team_rank({name},{i+1}).")
            
            # Handle host
            if custom_host.strip():
                if custom_host.strip() in team_names:
                    host_facts = [f"host({custom_host.strip()})."]
                else:
                    st.error(f"Host team '{custom_host}' not found in team list")
                    st.stop()
            else:
                random.seed(42)
                random_host = random.choice(team_names)
                host_facts = [f"host({random_host})."]
            
            # Generate groups
            group_facts = [f"group(g{i+1})." for i in range(num_teams // 4)]
            
            # Handle custom venues or generate default
            if custom_venues.strip():
                venue_names = [name.strip() for name in custom_venues.split(',')]
                if len(venue_names) != num_venues:
                    st.error(f"Please provide exactly {num_venues} venue names")
                    st.stop()
                venue_facts = [f"venue({name})." for name in venue_names]
            else:
                venue_names = [f"v{i+1}" for i in range(num_venues)]
                venue_facts = [f"venue({name})." for name in venue_names]
            
            # Generate venue capacities (using generator.py logic)
            capacity_facts = []
            for i, venue in enumerate(venue_names):
                tier = "large" if i < max(1, len(venue_names) // 2) else "standard"
                capacity_facts.append(f"venue_capacity({venue},{tier}).")
            
            # Generate time slots
            slot_facts = [f"day(1..{num_days}).", f"time_slot(1..{num_slots})."]
            
            # Combine all facts
            all_facts = (const_facts + team_facts+host_facts+rank_facts+confed_facts+group_facts+venue_facts+capacity_facts+slot_facts)
            
            # Add additional custom facts
            if additional_facts.strip():
                all_facts.append("\n% Additional Custom Facts")
                all_facts.extend(additional_facts.strip().split('\n'))
            
            instance_content = '\n'.join(all_facts)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Generated Instance")
                st.code(instance_content[:600] + "..." if len(instance_content) > 600 else instance_content, language="prolog")
            
            with col2:
                st.subheader("Download")
                filename = f"facts_{num_teams}teams.lp"
                st.download_button("Download .lp Instance File", data=instance_content, file_name=filename, mime="text/plain")
        
        except Exception as e:
            st.error(f"Error generating instance: {str(e)}")

st.divider()

# === RESULTS COMPARISON ===
st.subheader("📊 Results Comparison")

st.write("Compare Clingo and CP-SAT solver results by pasting outputs or uploading files.")

# Create two comparison columns
comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.write("**Result 1**")
    solver1_type = st.selectbox("Solver Type:", ["Clingo", "CP-SAT"], key="solver1_type")
    input1_method = st.radio("Input Method:", ["Paste", "Upload"], key="input1_method")
    
    if input1_method == "Paste":
        solver1_output = st.text_area(f"{solver1_type} Output", height=200,
            placeholder="Models: 5\nOptimum: yes\nOptimization: 42..." if solver1_type == "Clingo" else "Status: FEASIBLE\nSolutions found: 10\nTime: 12.456 seconds...",
            key="solver1_paste"
        )
    else:
        solver1_file = st.file_uploader(f"Upload {solver1_type} Result File", type=['txt'], key="solver1_file")
        solver1_output = solver1_file.read().decode('utf-8') if solver1_file else ""

with comparison_col2:
    st.write("**Result 2**")
    solver2_type = st.selectbox("Solver Type:", ["Clingo", "CP-SAT"], key="solver2_type")
    input2_method = st.radio("Input Method:", ["Paste", "Upload"], key="input2_method")
    
    if input2_method == "Paste":
        solver2_output = st.text_area(f"{solver2_type} Output", height=200,
            placeholder="Models: 5\nOptimum: yes\nOptimization: 42..." if solver2_type == "Clingo" else "Status: FEASIBLE\nSolutions found: 10\nTime: 12.456 seconds...",
            key="solver2_paste"
        )
    else:
        solver2_file = st.file_uploader(f"Upload {solver2_type} Result File", type=['txt'], key="solver2_file")
        solver2_output = solver2_file.read().decode('utf-8') if solver2_file else ""

if st.button("Compare Results", type="primary"):
    if not solver1_output and not solver2_output:
        st.warning("Please provide at least one solver output to analyze.")
    else:
        comparison_data = []
        
        if solver1_output:
            if solver1_type == "Clingo": solver1_data = parse_clingo_result(solver1_output)
            else: solver1_data = parse_cpsat_result(solver1_output)
            comparison_data.append(solver1_data)
        
        if solver2_output:
            if solver2_type == "Clingo":
                solver2_data = parse_clingo_result(solver2_output)
            else:
                solver2_data = parse_cpsat_result(solver2_output)
            comparison_data.append(solver2_data)
        
        if comparison_data:
            # Create comparison dataframe
            df = pd.DataFrame(comparison_data)
            
            # Display metrics
            st.subheader("Performance Comparison")
            
            # Key metrics table
            display_df = df.copy()
            if "solutions_found" in display_df.columns and "models" in display_df.columns:
                display_df["models"] = display_df["models"].fillna(display_df["solutions_found"])
                display_df = display_df.drop(columns=["solutions_found"])

            metrics_to_show = ['solver', 'status', 'solving_time', 'objective_value', 'models']
            available_metrics = [m for m in metrics_to_show if m in display_df.columns and not display_df[m].isna().all()]

            if available_metrics:
                st.dataframe(display_df[available_metrics], use_container_width=True)

            # Display comparison stats
            if len(comparison_data) == 2:
                data1, data2 = comparison_data
                col1, col2 = st.columns(2)
                with col1:
                    if data1.get('solving_time') and data2.get('solving_time'):
                        st.metric("Solving Time Difference", f"{abs(data1['solving_time'] - data2['solving_time']):.2f}s")
                
                with col2:
                    if data1.get('objective_value') and data2.get('objective_value'):
                        diff = data1['objective_value'] - data2['objective_value']
                        st.metric("Objective Value Difference", f"{abs(diff)}")
                
                solutions1 = data1.get('models') or data1.get('solutions_found', 0)
                solutions2 = data2.get('models') or data2.get('solutions_found', 0)
                if solutions1 and solutions2:
                    st.metric("Solutions Found", f"{data1['solver']}: {solutions1}, {data2['solver']}: {solutions2}")


st.divider()

# ==== PARSE AND DISPLAY SCHEDULE =====
st.subheader("🗓️ Schedule Visualisation")

schedule_facts = ""
schedule_facts = st.text_area("Paste an output schedule facts here:", height=300,
    placeholder="assigned_group(t1,g1,1).\nscheduled_match(t1,t2,g1,1,1,v1,1).\n...",
    help="Paste the assigned_group and scheduled_match facts from Clingo output"
)


if st.button("Display Formatted Schedule") and schedule_facts:
    try:
        # Parse facts
        match_pattern = re.compile(r"scheduled_match\((\w+),(\w+),(\w+),(\d+),(\d+),(\w+),(\d+)\)")
        group_pattern = re.compile(r"assigned_group\((\w+),(\w+),(\d+)\)")

        matches_data = [{
                "Team 1": m[0], "Team 2": m[1], "Group": m[2], 
                "Day": int(m[3]), "Slot": int(m[4]), "Venue": m[5], "Round": int(m[6])
                }
            for m in match_pattern.findall(schedule_facts)
        ]
        
        groups_data = [{"Team": g[0], "Group": g[1], "Position": int(g[2])} for g in group_pattern.findall(schedule_facts)]

        if not matches_data and not groups_data:
            st.warning("No valid schedule facts found in input.")
        else:
            st.success(f"Parsed {len(matches_data)} matches and {len(groups_data)} group assignments")
            # Create dataframes
            groups_df = pd.DataFrame(groups_data)
            matches_df = pd.DataFrame(matches_data)

            # Display tabs
            tab_overview, tab_daily, tab_groups = st.tabs(["📊 Overview", "📅 Daily Schedule", "🏆 Groups"])

            with tab_overview:
                if not matches_df.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Matches", len(matches_df))
                    with col2: st.metric("Tournament Days", matches_df['Day'].max())
                    with col3: st.metric("Groups", matches_df['Group'].nunique())
                    with col4: st.metric("Venues Used", matches_df['Venue'].nunique())
                    
                    # Match distribution chart
                    st.subheader("Matches per Day")
                    daily_counts = matches_df.groupby('Day').size()
                    st.bar_chart(daily_counts)
                    
                    # Venue utilisation
                    st.subheader("Venue Utilization")
                    venue_counts = matches_df.groupby('Venue').size()
                    st.bar_chart(venue_counts)

                    # Time slot distribution
                    st.markdown("**Time Slot Usage:**")
                    slot_usage = matches_df['Slot'].value_counts().sort_index()
                    st.bar_chart(slot_usage)

            with tab_daily:
                st.subheader("Daily Schedule")
                if not matches_df.empty:
                    sorted_matches = matches_df.sort_values(by=["Day", "Slot"])
                    
                    for day, day_schedule in sorted_matches.groupby("Day"):
                        with st.container(border=True):
                            st.markdown(f"#### Day {day}")
                            for slot, slot_matches in day_schedule.groupby("Slot"):
                                for _, match in slot_matches.iterrows():
                                    st.markdown(
                                        f"`{map_group(match['Group'])}` | {map_slot(slot)} | **{format_word(match['Team 1']).capitalize()}** vs **{format_word(match['Team 2']).capitalize()}** | Venue: *{format_word(match['Venue']).capitalize()}*"
                                    )
                                st.write("")
                else:
                    st.info("No match schedule found.")

                # Export options
                st.subheader("Export Schedule")
                if not matches_df.empty:
                    csv_data = matches_df.to_csv(index=False)
                    st.download_button(
                        "Download as CSV",
                        data=csv_data,
                        file_name=f"tournament_schedule.csv",
                        mime="text/csv"
                    )

            with tab_groups:
                st.subheader("Groups")
                if not groups_df.empty:
                    groups_list = sorted(groups_df["Group"].unique())
                    num_cols = min(len(groups_list), 4)
                    cols = st.columns(num_cols)
                    
                    for idx, group in enumerate(groups_list):
                        with cols[idx % num_cols]:
                            with st.container(border=True):
                                st.markdown(f"**{map_group(group)}**")
                                group_teams = groups_df[groups_df["Group"] == group].sort_values("Position")
                                for _, team_info in group_teams.iterrows():
                                    st.markdown(f"{team_info['Position']}. {format_word(team_info['Team']).capitalize()}")
                else:
                    st.info("No group assignments found.")

    except Exception as e:
        st.error(f"Error parsing schedule: {str(e)}")
        st.write("Please check the input contains valid ASP facts.")

st.divider()