"""
Cleanup script to close unused Streamlit ports
Run this when dashboards have issues starting
"""

import subprocess
import sys

def cleanup_streamlit_ports():
    """Kill all Streamlit processes to free up ports."""
    print("🧹 Cleaning up Streamlit ports...")
    
    try:
        # Get all python processes
        result = subprocess.run([
            'powershell', '-Command',
            'Get-Process | Where-Object {$_.ProcessName -eq "python"} | Select-Object Id, ProcessName'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            process_ids = []
            
            for line in lines[2:]:  # Skip header lines
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            pid = int(parts[0])
                            process_ids.append(pid)
                        except ValueError:
                            continue
            
            # Check which processes are using Streamlit ports
            streamlit_pids = []
            for pid in process_ids:
                try:
                    # Check if process is using Streamlit ports (850x)
                    netstat_result = subprocess.run([
                        'netstat', '-ano'
                    ], capture_output=True, text=True)
                    
                    if f"{pid}" in netstat_result.stdout and ":850" in netstat_result.stdout:
                        streamlit_pids.append(pid)
                except:
                    continue
            
            # Kill Streamlit processes
            for pid in streamlit_pids:
                try:
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                 capture_output=True, check=False)
                    print(f"✅ Killed process {pid}")
                except:
                    print(f"❌ Failed to kill process {pid}")
            
            if streamlit_pids:
                print(f"🎯 Cleaned up {len(streamlit_pids)} Streamlit processes")
            else:
                print("✅ No Streamlit processes found to clean up")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def check_port_status():
    """Check which ports are currently in use."""
    print("\n📊 Current port status:")
    
    try:
        result = subprocess.run([
            'netstat', '-ano'
        ], capture_output=True, text=True)
        
        lines = result.stdout.split('\n')
        streamlit_ports = []
        
        for line in lines:
            if ':850' in line and 'LISTENING' in line:
                streamlit_ports.append(line.strip())
        
        if streamlit_ports:
            print("🔍 Active Streamlit ports:")
            for port_line in streamlit_ports:
                print(f"  {port_line}")
        else:
            print("✅ No Streamlit ports in use")
    
    except Exception as e:
        print(f"❌ Error checking ports: {e}")

if __name__ == "__main__":
    cleanup_streamlit_ports()
    check_port_status()
    print("\n🎯 Port cleanup complete!")
    print("You can now start a new dashboard without port conflicts.")