with open('../setup.sh', 'w') as f:
    f.write('mkdir -p ~/.streamlit/\n')
    f.write('echo "[server]" > ~/.streamlit/config.toml\n')
    f.write('echo "headless = true" >> ~/.streamlit/config.toml\n')
    f.write('echo "port = $PORT" >> ~/.streamlit/config.toml\n')
    f.write('echo "enableCORS = false" >> ~/.streamlit/config.toml\n')

print("setup.sh fixed!")

with open('../setup.sh', 'r') as f:
    print(f.read())