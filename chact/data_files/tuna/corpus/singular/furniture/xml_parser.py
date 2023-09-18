import xml.etree.ElementTree as ET

# Load the XML file:
tree = ET.parse("s4t1.xml")
root = tree.getroot()

instances = []

for child in root:
	print(child.tag, child.attrib)

for entity in root.findall(".//ENTITY"):
	entity_id = entity.get("ID")  # id of the entity
	entity_image = entity.get("IMAGE")  # the file name of the image
	entity_type = entity.get("TYPE")  # target/distractor
	# print(entity_type)

	instance = {}
	instance['id'] = entity_id
	instance['classification'] = entity_type
	# print(instance)

    # Iterate through ATTRIBUTE elements for this ENTITY
	for attribute in entity.findall(".//ATTRIBUTE"):
		attribute_name = attribute.get("NAME")
		attribute_type = attribute.get("TYPE")
		attribute_value = attribute.get("VALUE")

		print(f"Entity ID: {entity_id}")
		print(f"Entity Image: {entity_image}")
		print(f"Entity Type: {entity_type}")
		print(f"Attribute Name: {attribute_name}")
		print(f"Attribute Type: {attribute_type}")
		print(f"Attribute Value: {attribute_value}")
		print("--------------------")

		instance[attribute_name] = attribute_value
    
	instances.append(instance)

for instance in instances:
	print(instance)
