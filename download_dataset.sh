if [ "$#" -eq 0 ]; then
    OUTPUT_DIR="data/celeba"
    echo "No output directory specified. Using default: ${OUTPUT_DIR}"
else
    OUTPUT_DIR=$1
    echo "Output directory specified: ${OUTPUT_DIR}"
fi


if [ -d "${OUTPUT_DIR}" ]; then
    echo "Directory ${OUTPUT_DIR} already exists."
else
    echo "Creating directory ${OUTPUT_DIR}..."
    mkdir -p ${OUTPUT_DIR}
fi


if [ -d "${OUTPUT_DIR}/img_align_celeba" ]; then
    echo "CelebA dataset already exists."
    exit 0
else
    echo "Downloading CelebA dataset..."
    curl -L -o ${OUTPUT_DIR}/celeba-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
    echo "Download completed."
    
    echo "Unzipping CelebA dataset..."
    unzip -q ${OUTPUT_DIR}/celeba-dataset.zip -d ${OUTPUT_DIR}
    echo "Unzipping completed."

    echo "Deleting the zip file..."
    rm ${OUTPUT_DIR}/celeba-dataset.zip
    echo "Zip file deleted."
fi
