./power power_output.csv &
PID=$!

sleep 10

kill $PID
