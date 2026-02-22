from util import *


# Test
sample_plate_chars = "ABC1234"
sample_plate_img = generate_license_plate_image(sample_plate_chars)
sample_plate_img.save("test_plate.png")
print("Predicted generated plate:", predict_plate_chars("test_plate.png") , f"Actual plate: {sample_plate_chars}")

print("EU")
print("Predicted real plate:", predict_plate_chars("./real_image/EU/image1.png"), f"Actual plate: 7995FJH")
print("Predicted real plate:", predict_plate_chars("./real_image/EU/image2.png"), f"Actual plate: 4370BCS")
print("Predicted real plate:", predict_plate_chars("./real_image/EU/image3.png"), f"Actual plate: KRT 290")
print("Predicted real plate:", predict_plate_chars("./real_image/EU/image4.png"), f"Actual plate: 4671CCC")
print("Predicted real plate:", predict_plate_chars("./real_image/EU/image5.png"), f"Actual plate: 14BJN83")


print("International")
print("Predicted real plate:", predict_plate_chars("./real_image/International/image1.png"), f"Actual plate: PBANEGO")
print("Predicted real plate:", predict_plate_chars("./real_image/International/image2.png"), f"Actual plate: BCA9G35")
print("Predicted real plate:", predict_plate_chars("./real_image/International/image3.png"), f"Actual plate: OKKAR3N")
print("Predicted real plate:", predict_plate_chars("./real_image/International/image4.png"), f"Actual plate: W-066901")
print("Predicted real plate:", predict_plate_chars("./real_image/International/image5.png"), f"Actual plate: WÜ04784")