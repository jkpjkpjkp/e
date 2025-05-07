
        # Get the arguments and ensure they're in dictionary format
        args = response.tool_calls[0].function.arguments
        if isinstance(args, str):
            args = json.loads(args)

        # Make sure image is included in the arguments and it's a PIL Image
        if 'image' not in args:
            args['image'] = image
        elif isinstance(args['image'], str):
            # If image is a URL, download it and convert to PIL Image
            try:
                response = requests.get(args['image'])
                args['image'] = Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Error loading image from URL: {e}")
                args['image'] = image

        tool_response = globals()[response.tool_calls[0].function.name](**args)