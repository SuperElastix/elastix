
#ifndef __itkAdvancedMeanSquaresImageToImageMetric_h
#define __itkAdvancedMeanSquaresImageToImageMetric_h

#include "itkImageToImageMetricWithSampling.h"
#include "itkDerivativeOperator.h"


namespace itk
{
/** \class AdvancedMeanSquaresImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metric computes the sum of squared differenced between pixels in
 * the moving image and pixels in the fixed image. The spatial correspondance 
 * between both images is established through a Transform. Pixel values are
 * taken from the Moving image. Their positions are mapped to the Fixed image
 * and result in general in non-grid position on it. Values at these non-grid
 * position of the Fixed image are interpolated using a user-selected Interpolator.
 *
 * This class provides functionality to calculate (the derivative of) the
 * mean squares metric on only a subset of the fixed image voxels. This
 * option is controlled by the boolean UseAllPixels, which is by default true.
 * Substantial speedup can be accomplished by setting it to false and specifying
 * the NumberOfSpacialSamples to some small portion of the total number of fixed
 * image samples. The samples are randomly chosen using an
 * itk::ImageRandomConstIteratorWithIndex Every iteration a new set of those
 * samples are used. This is important, because the error made by calculating
 * the metric value with only a subset of all samples should be randomly
 * distributed with zero mean.
 *
 * \todo In the while loop in GetValue and GetValueAndDerivative another for
 * loop is made over all parameters. In case of a B-spline transform advantage
 * can be taken from the fact that it has compact support, similar to the
 * itk::MattesMutualInformationImageToImageMetric.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */
template < class TFixedImage, class TMovingImage > 
class AdvancedMeanSquaresImageToImageMetric : 
    public ImageToImageMetricWithSampling< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef AdvancedMeanSquaresImageToImageMetric		Self;
  typedef ImageToImageMetricWithSampling<
    TFixedImage, TMovingImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedMeanSquaresImageToImageMetric, ImageToImageMetricWithSampling );

  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;
  typedef typename Superclass::InputPointType			      InputPointType;
  typedef typename Superclass::OutputPointType		      OutputPointType;
  typedef typename Superclass::ImageSamplerType         ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer      ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType      ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer   ImageSampleContainerPointer;
  
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  
	/** The fixed image dimension. */
	itkStaticConstMacro( FixedImageDimension, unsigned int,
		FixedImageType::ImageDimension );

	/** The moving image dimension. */
	itkStaticConstMacro( MovingImageDimension, unsigned int,
		MovingImageType::ImageDimension );

  typedef typename MovingImageType::SpacingType           MovingImageSpacingType;
  typedef typename MovingImageMaskType::OutputVectorType  MovingMaskDerivativeType;
  typedef MovingImageSpacingType                          MovingRealOffsetType;
  typedef itk::Neighborhood<
    MovingRealOffsetType, 
    itkGetStaticConstMacro(MovingImageDimension)>         MovingMaskNeighborhoodOffsetsType;
  typedef itk::Neighborhood<
    double, 
    itkGetStaticConstMacro(MovingImageDimension)>         MovingMaskDerivativeOperatorType;
  typedef itk::FixedArray<
    MovingMaskNeighborhoodOffsetsType>                    MovingMaskNeighborhoodOffsetsArrayType;
  typedef itk::FixedArray<
    MovingMaskDerivativeOperatorType>                     MovingMaskDerivativeOperatorArrayType;
  typedef itk::DerivativeOperator<
    double, 
    itkGetStaticConstMacro(MovingImageDimension)>         DefaultMovingMaskDerivativeOperatorType;
  

	/** Get the value for single valued optimizers. */
	virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  virtual void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivative( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li set the internal moving mask
   * \li set the moving mask derivative scales
   */
  virtual void Initialize(void) throw ( ExceptionObject );

  /** Get the internal moving image mask. Equals the movingimage mask if set, and 
   * otherwise it's a box with size equal to the moving image's largest possible region */
  itkGetConstObjectMacro(InternalMovingImageMask, MovingImageMaskType);

  /** Set an operator used to take the derivative of the moving mask;
   * for each dimension
   * Watch out, in the Initialize method the derivative operators are 
   * replaced by the default operator, currently...
   * This is maybe not really convenient.
   */
  virtual void SetMovingMaskDerivativeOperator(unsigned int dim, 
    const MovingMaskDerivativeOperatorType & op)
  { 
    this->m_MovingMaskDerivativeOperatorArray[dim] = op;
  };

  /** Get the operator used to take the derivative of the moving mask;
   * for each dimension */
  virtual const MovingMaskDerivativeOperatorType & GetMovingMaskDerivativeOperator(
    unsigned int dim) const
  {
    return this->m_MovingMaskDerivativeOperatorArray[dim];
  };

protected:
  AdvancedMeanSquaresImageToImageMetric();
  virtual ~AdvancedMeanSquaresImageToImageMetric() {};
	void PrintSelf( std::ostream& os, Indent indent ) const;

  /** A copy of the mask, that is set to a boxspatialobject if the 
   * user has not entered any mask */
  MovingImageMaskPointer m_InternalMovingImageMask;

  /** The operators that are used to take the derivative of the moving mask */
  MovingMaskDerivativeOperatorArrayType   m_MovingMaskDerivativeOperatorArray;
  MovingMaskNeighborhoodOffsetsArrayType  m_MovingMaskNeighborhoodOffsetsArray;
  
  /** Estimate value and spatial derivative of internal moving mask */
  virtual void EvaluateMovingMaskValueAndDerivative(
    const OutputPointType & point,
    double & value,
    MovingMaskDerivativeType & derivative) const;

  /** Functions called from Initialize, to split up that function a bit. */
  virtual void InitializeDerivativeOperators(void);
  virtual void InitializeInternalMasks(void);
  virtual void InitializeNeighborhoodOffsets(void);


private:
  AdvancedMeanSquaresImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class AdvancedMeanSquaresImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedMeanSquaresImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkAdvancedMeanSquaresImageToImageMetric_h

