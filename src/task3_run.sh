for dir in quizType*; do
    if [ -d "$dir" ]; then
	python3 task3.py "$dir" > "$dir".csv
    fi
done
