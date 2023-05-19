source /home/user/anaconda3/etc/profile.d/conda.sh
# DIR="/media/mdai/Data_House_3/projects/api-2dgpn"

conda deactivate

conda activate sd2-api

uvicorn api:app \
--host 0.0.0.0 \
--port 7003 \
--workers 1 >sd2-api.log 2>&1 &