## ```BaseImageProcessorFast```

```python
class BaseImageProcessorFast(BaseImageProcessor):
    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_pad = None
    pad_size = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    return_tensors = None
    data_format = ChannelDimension.FIRST
    input_data_format = None
    device = None
    model_input_names = ["pixel_values"]
    valid_kwargs = DefaultFastImageProcessorKwargs
    unused_kwargs = None
```

## ```preprocess```

```
    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        # args are not validated, but their order in the `preprocess` and `_preprocess` signatures must be the same
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_kwargs_names)
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")

        return self._preprocess_image_like_inputs(
            images, *args, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )
```
