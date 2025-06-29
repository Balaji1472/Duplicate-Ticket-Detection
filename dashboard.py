"""
Simple Streamlit Dashboard for Duplicate Ticket Detection API
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from typing import Dict, List

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Duplicate Ticket Detection Dashboard",
    page_icon="🎫",
    layout="wide"
)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def check_duplicate(ticket_id: str, text: str, language: str = None):
    """Check for duplicate ticket"""
    try:
        data = {"ticket_id": ticket_id, "text": text}
        if language:
            data["language"] = language
        
        response = requests.post(f"{API_BASE_URL}/check-duplicate", json=data)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def add_ticket(ticket_id: str, text: str, language: str = None):
    """Add new ticket"""
    try:
        data = {"ticket_id": ticket_id, "text": text}
        if language:
            data["language"] = language
        
        response = requests.post(f"{API_BASE_URL}/add-ticket", json=data)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def submit_feedback(ticket_id: str, matched_ticket_id: str, is_correct: bool, comment: str = ""):
    """Submit feedback"""
    try:
        data = {
            "ticket_id": ticket_id,
            "matched_ticket_id": matched_ticket_id,
            "is_correct_duplicate": is_correct,
            "user_comment": comment
        }
        
        response = requests.post(f"{API_BASE_URL}/feedback", json=data)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_tickets():
    """Get all tickets"""
    try:
        response = requests.get(f"{API_BASE_URL}/tickets?limit=1000")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main Dashboard
st.title("🎫 Duplicate Ticket Detection Dashboard")

# Check API status
if not check_api_health():
    st.error("❌ API is not running. Please start the FastAPI server first.")
    st.code("python main.py")
    st.stop()

st.success("✅ API is running")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Overview", 
    "Check Duplicates", 
    "Add Ticket", 
    "View Tickets",
    "Submit Feedback"
])

# Overview Page
if page == "Overview":
    st.header("📊 System Overview")
    
    stats = get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickets", stats.get('total_tickets', 0))
        
        with col2:
            st.metric("Model", stats.get('model_name', 'N/A').split('/')[-1])
        
        with col3:
            st.metric("Threshold", f"{stats.get('similarity_threshold', 0):.2f}")
        
        with col4:
            feedback_stats = stats.get('feedback_stats', {})
            st.metric("Feedback Count", feedback_stats.get('total_feedback', 0))
        
        # Language distribution
        lang_dist = stats.get('language_distribution', [])
        if lang_dist:
            st.subheader("🌍 Language Distribution")
            df = pd.DataFrame(lang_dist)
            if not df.empty:
                fig = px.pie(df, values='ticket_count', names='language_name', 
                           title="Tickets by Language")
                st.plotly_chart(fig, use_container_width=True)
        
        # Features status
        st.subheader("🔧 Features Status")
        features = stats.get('features_enabled', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅" if features.get('multilingual_support') else "❌"
            st.write(f"{status} Multilingual Support")
        
        with col2:
            status = "✅" if features.get('clustering') else "❌"
            st.write(f"{status} Clustering")
        
        with col3:
            status = "✅" if features.get('feedback_collection') else "❌"
            st.write(f"{status} Feedback Collection")

# Check Duplicates Page
elif page == "Check Duplicates":
    st.header("🔍 Check for Duplicates")
    
    with st.form("duplicate_check_form"):
        ticket_id = st.text_input("Ticket ID", placeholder="e.g., TKT-001")
        text = st.text_area("Ticket Description", placeholder="Enter the ticket description...")
        language = st.selectbox("Language (optional)", 
                               ["Auto-detect", "en", "es", "fr", "de", "it", "pt", "hi", "zh","ta"])
        
        submitted = st.form_submit_button("Check Duplicates")
        
        if submitted and ticket_id and text:
            lang = None if language == "Auto-detect" else language
            
            with st.spinner("Checking for duplicates..."):
                result = check_duplicate(ticket_id, text, lang)
            
            if result:
                st.subheader("Results")
                
                if result['is_duplicate']:
                    st.warning(f"⚠️ Potential duplicates found!")
                    
                    for i, match in enumerate(result['matches'], 1):
                        with st.expander(f"Match #{i} - Similarity: {match['similarity_score']:.3f}"):
                            st.write(f"**Ticket ID:** {match['ticket_id']}")
                            st.write(f"**Text:** {match['text']}")
                            st.write(f"**Language:** {match.get('language', 'Unknown')}")
                else:
                    st.success("✅ No duplicates found!")
                
                # Additional info
                st.subheader("Additional Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Detected Language:** {result.get('detected_language', 'Unknown')}")
                
                with col2:
                    cluster_id = result.get('cluster_id')
                    st.write(f"**Cluster ID:** {cluster_id if cluster_id is not None else 'None'}")
            else:
                st.error("Failed to check duplicates. Please try again.")

# Add Ticket Page
elif page == "Add Ticket":
    st.header("➕ Add New Ticket")
    
    with st.form("add_ticket_form"):
        ticket_id = st.text_input("Ticket ID", placeholder="e.g., TKT-002")
        text = st.text_area("Ticket Description", placeholder="Enter the ticket description...")
        language = st.selectbox("Language (optional)", 
                               ["Auto-detect", "en", "es", "fr", "de", "it", "pt", "hi", "zh"])
        
        submitted = st.form_submit_button("Add Ticket")
        
        if submitted and ticket_id and text:
            lang = None if language == "Auto-detect" else language
            
            with st.spinner("Adding ticket..."):
                result = add_ticket(ticket_id, text, lang)
            
            if result:
                st.success(f"✅ {result['message']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Total Tickets:** {result['total_tickets']}")
                with col2:
                    st.write(f"**Detected Language:** {result.get('detected_language', 'Unknown')}")
                with col3:
                    cluster_id = result.get('cluster_id')
                    st.write(f"**Cluster ID:** {cluster_id if cluster_id is not None else 'None'}")
            else:
                st.error("Failed to add ticket. It might already exist or there was an error.")

# View Tickets Page
elif page == "View Tickets":
    st.header("📋 View All Tickets")
    
    tickets_data = get_tickets()
    
    if tickets_data and tickets_data['tickets']:
        st.write(f"Showing {tickets_data['returned_count']} of {tickets_data['total_count']} tickets")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(tickets_data['tickets'])
        
        # Add search functionality
        search_term = st.text_input("🔍 Search tickets", placeholder="Search by ID or text...")
        
        if search_term:
            mask = df['ticket_id'].str.contains(search_term, case=False, na=False) | \
                   df['text'].str.contains(search_term, case=False, na=False)
            df = df[mask]
        
        # Display tickets
        for _, ticket in df.iterrows():
            with st.expander(f"Ticket: {ticket['ticket_id']} ({ticket.get('language', 'Unknown')})"):
                st.write(f"**Text:** {ticket['text']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Language:** {ticket.get('language', 'Unknown')}")
                with col2:
                    cluster_id = ticket.get('cluster_id')
                    st.write(f"**Cluster:** {cluster_id if cluster_id is not None else 'None'}")
    else:
        st.info("No tickets found.")

# Submit Feedback Page
elif page == "Submit Feedback":
    st.header("💭 Submit Feedback")
    
    st.info("Help improve the system by providing feedback on duplicate detection results.")
    
    with st.form("feedback_form"):
        ticket_id = st.text_input("Original Ticket ID", placeholder="e.g., TKT-001")
        matched_ticket_id = st.text_input("Matched Ticket ID (if any)", placeholder="e.g., TKT-002")
        
        is_correct = st.radio(
            "Was the duplicate detection correct?",
            ["Yes, it was correct", "No, it was incorrect"]
        )
        
        comment = st.text_area("Additional Comments (optional)", 
                              placeholder="Any additional feedback...")
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted and ticket_id:
            is_correct_bool = is_correct.startswith("Yes")
            
            with st.spinner("Submitting feedback..."):
                result = submit_feedback(ticket_id, matched_ticket_id or None, 
                                       is_correct_bool, comment)
            
            if result:
                st.success(f"✅ {result['message']}")
                st.write(f"Feedback ID: {result['feedback_id']}")
                st.write(f"Total feedback collected: {result['total_feedback_count']}")
            else:
                st.error("Failed to submit feedback. Please try again.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🎫 Duplicate Ticket Detection**")
st.sidebar.markdown("Simple dashboard for API interaction")
