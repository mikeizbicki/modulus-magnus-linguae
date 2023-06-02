for dir in quizType*; do
    if [ -d "$dir" ]; then
	python3 lmqlToAccuracyNum.py "$dir"
    fi
done
