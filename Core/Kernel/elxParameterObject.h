// ...existing code...

// Ensure FixedImageType and MovingImageType are defined before using MaskImageType
using FixedImageType = itk::Image<float, 3>; // Example definition, adjust as necessary
using MovingImageType = itk::Image<float, 3>; // Example definition, adjust as necessary

namespace elastix
{
// ...existing code...

  /** Types for the masks. */
  using MaskPixelType = unsigned char;
  using MaskImageType = itk::Image<MaskPixelType, itk::GetImageDimension<FixedImageType>::ImageDimension>;

// ...existing code...
};
} // end namespace elastix

#endif // end #ifndef elxParameterObject_h
