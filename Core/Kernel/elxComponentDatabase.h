#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkObject.h"
// Include the headers where FixedImageType and MovingImageType are defined
// If they are not defined elsewhere, define them here
// #include "itkFixedImageType.h"
// #include "itkMovingImageType.h"

// Ensure FixedImageType and MovingImageType are defined before using MaskImageType
using FixedImageType = itk::Image<float, 3>; // Example definition, adjust as necessary
using MovingImageType = itk::Image<float, 3>; // Example definition, adjust as necessary

namespace elastix
{

class ComponentDatabase : public itk::Object
{
public:

  /** Types for the masks. */
  using MaskPixelType = unsigned char;
  // Ensure MaskImageType is defined before using it
  using MaskImageType = itk::Image<MaskPixelType, itk::GetImageDimension<FixedImageType>::ImageDimension>;

};
} // end namespace elastix

#endif // end #ifndef elxComponentDatabase_h
