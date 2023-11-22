#echo $NUM_WORKERS
for (( i=0; i<$NUM_WORKERS; i++ ))
do
   I_GID=$(($RUN_GID+$i))
   I_UID=$(($RUN_UID+$i))
   groupadd -g $I_GID runner$I_GID && useradd -M runner$I_UID -g $I_GID -u $I_UID
   echo "Created " $(id runner$I_UID)
done

gunicorn \
    -w ${NUM_WORKERS} \
    --bind 0.0.0.0:${GUNICORN_PORT} \
    --timeout 0 \
    --log-level ${LOG_LEVEL} \
    "wsgi:app"
