/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkTransformToSpatialJacobianSource_h
#define itkTransformToSpatialJacobianSource_h

#include "itkAdvancedTransform.h"
#include "itkAdvancedIdentityTransform.h"
#include "itkImageSource.h"

namespace itk
{

/** \class TransformToSpatialJacobianSource
 * \brief Generate the spatial Jacobian matrix from a coordinate transform
 *
 * This class was inspired on an the
 * itkTransformToDeterminantOfSpatialJacobianSource class.
 * It generates the full spatial jacobian matrix. The output image
 * type should thus be an image with a matrix-pixeltype, e.g., the
 * SpatialJacobianType defined in the itkAdvancedTransform class.
 * Suggested/recommended choices are the
 * itk::Matrix<double, ImageDimension, ImageDimension> (=SpatialJacobianType)
 * or
 * itk::Matrix<float, ImageDimension, ImageDimension>
 *
 * Output information (spacing, size and direction) for the output
 * image should be set. This information has the normal defaults of
 * unit spacing, zero origin and identity direction. Optionally, the
 * output information can be obtained from a reference image. If the
 * reference image is provided and UseReferenceImage is On, then the
 * spacing, origin and direction of the reference image will be used.
 *
 * Since this filter produces an image which is a different size than
 * its input, it needs to override several of the methods defined
 * in ProcessObject in order to properly manage the pipeline execution model.
 * In particular, this filter overrides
 * ProcessObject::GenerateInputRequestedRegion() and
 * ProcessObject::GenerateOutputInformation().
 *
 * This filter is implemented as a multithreaded filter.  It provides a
 * ThreadedGenerateData() method for its implementation.
 *
 * \author Stefan Klein, Erasmus MC, The Netherlands.
 *
 * This class was taken from the Insight Journal paper:
 * http://hdl.handle.net/?/?
 *
 *
 * \ingroup GeometricTransforms
 */
template <class TOutputImage, class TTransformPrecisionType = double>
class ITK_TEMPLATE_EXPORT TransformToSpatialJacobianSource : public ImageSource<TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformToSpatialJacobianSource);

  /** Standard class typedefs. */
  using Self = TransformToSpatialJacobianSource;
  using Superclass = ImageSource<TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using OutputImageConstPointer = typename OutputImageType::ConstPointer;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformToSpatialJacobianSource, ImageSource);

  /** Number of dimensions. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Typedefs for transform. */
  using TransformType = AdvancedTransform<TTransformPrecisionType, Self::ImageDimension, Self::ImageDimension>;
  using TransformPointerType = typename TransformType::ConstPointer;
  using SpatialJacobianType = typename TransformType::SpatialJacobianType;

  /** Typedefs for output image. */
  using PixelType = typename OutputImageType::PixelType;
  // typedef typename PixelType::ValueType           PixelValueType;
  using RegionType = typename OutputImageType::RegionType;
  using SizeType = typename RegionType::SizeType;
  using IndexType = typename OutputImageType::IndexType;
  using PointType = typename OutputImageType::PointType;
  using SpacingType = typename OutputImageType::SpacingType;
  using OriginType = typename OutputImageType::PointType;
  using DirectionType = typename OutputImageType::DirectionType;

  /** Typedefs for base image. */
  using ImageBaseType = ImageBase<Self::ImageDimension>;

  /** Set the coordinate transformation.
   * Set the coordinate transform to use for resampling.  Note that this must
   * be in physical coordinates and it is the output-to-input transform, NOT
   * the input-to-output transform that you might naively expect.  By default
   * the filter uses an Identity transform. You must provide a different
   * transform here, before attempting to run the filter, if you do not want to
   * use the default Identity transform. */
  itkSetConstObjectMacro(Transform, TransformType);

  /** Get a pointer to the coordinate transform. */
  itkGetConstObjectMacro(Transform, TransformType);

  /** Set the size of the output image. */
  virtual void
  SetOutputSize(const SizeType & size);

  /** Get the size of the output image. */
  virtual const SizeType &
  GetOutputSize();

  /** Set the start index of the output largest possible region.
   * The default is an index of all zeros. */
  virtual void
  SetOutputIndex(const IndexType & index);

  /** Get the start index of the output largest possible region. */
  virtual const IndexType &
  GetOutputIndex();

  /** Set the region of the output image. */
  itkSetMacro(OutputRegion, OutputImageRegionType);

  /** Get the region of the output image. */
  itkGetConstReferenceMacro(OutputRegion, OutputImageRegionType);

  /** Set the output image spacing. */
  itkSetMacro(OutputSpacing, SpacingType);
  virtual void
  SetOutputSpacing(const double * values);

  /** Get the output image spacing. */
  itkGetConstReferenceMacro(OutputSpacing, SpacingType);

  /** Set the output image origin. */
  itkSetMacro(OutputOrigin, OriginType);
  virtual void
  SetOutputOrigin(const double * values);

  /** Get the output image origin. */
  itkGetConstReferenceMacro(OutputOrigin, OriginType);

  /** Set the output direction cosine matrix. */
  itkSetMacro(OutputDirection, DirectionType);
  itkGetConstReferenceMacro(OutputDirection, DirectionType);

  /** Helper method to set the output parameters based on this image */
  void
  SetOutputParametersFromImage(const ImageBaseType * image);

  /** TransformToSpatialJacobianSource produces a floating value image. */
  void
  GenerateOutputInformation() override;

  /** Checking if transform is set. If a linear transformation is used,
   * the unthreaded LinearGenerateData is called. */
  void
  BeforeThreadedGenerateData() override;

  /** Compute the Modified Time based on changes to the components. */
  ModifiedTimeType
  GetMTime() const override;

protected:
  TransformToSpatialJacobianSource();
  ~TransformToSpatialJacobianSource() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** TransformToSpatialJacobianSource can be implemented as a multithreaded
   * filter.
   */
  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

  /** Default implementation for resampling that works for any
   * transformation type.
   */
  void
  NonlinearThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId);

  /** Faster implementation for resampling that works for with linear
   *  transformation types. Unthreaded.
   */
  void
  LinearGenerateData();

private:
  /** Member variables. */
  RegionType           m_OutputRegion{}; // region of the output image
  TransformPointerType m_Transform{
    AdvancedIdentityTransform<TTransformPrecisionType, ImageDimension>::New()
  };                                                               // Coordinate transform to use
  SpacingType   m_OutputSpacing{ 1.0 };                            // output image spacing
  OriginType    m_OutputOrigin{};                                  // output image origin
  DirectionType m_OutputDirection{ DirectionType::GetIdentity() }; // output image direction cosines
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformToSpatialJacobianSource.hxx"
#endif

#endif // end #ifndef itkTransformToSpatialJacobianSource_h
