mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = '#667eea'\n\
backgroundColor = '#ffffff'\n\
secondaryBackgroundColor = '#f0f2f6'\n\
textColor = '#262730'\n\
" > ~/.streamlit/config.toml
