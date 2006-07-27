#ifndef _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx
#define _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx

#include "itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric.h"


namespace itk
{

  /**
	 * ************************ Constructor *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric()
  {
    this->SetComputeGradient(false); // don't use the default gradient for now
    this->m_NumberOfParameters = 0;
    this->m_Alpha = 0.5;

    this->m_BinaryKNNTreeFixedIntensity = 0;
    this->m_BinaryKNNTreeMovingIntensity = 0;
    this->m_BinaryKNNTreeJointIntensity = 0;
    
    this->m_BinaryKNNTreeSearcherFixedIntensity = 0;
    this->m_BinaryKNNTreeSearcherMovingIntensity = 0;
    this->m_BinaryKNNTreeSearcherJointIntensity = 0;

  } // end Constructor


  /**
	 * ************************ SetANNkDTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNkDTree( unsigned int bucketSize = 2, std::string splittingRule = "ANN_KD_SL_MIDPT" )
  {
    typename ANNkDTreeType::Pointer tmpPtrF = ANNkDTreeType::New();
    typename ANNkDTreeType::Pointer tmpPtrM = ANNkDTreeType::New();
    typename ANNkDTreeType::Pointer tmpPtrJ = ANNkDTreeType::New();
    
    tmpPtrF->SetBucketSize( bucketSize );
    tmpPtrM->SetBucketSize( bucketSize );
    tmpPtrJ->SetBucketSize( bucketSize );

    tmpPtrF->SetSplittingRule( splittingRule );
    tmpPtrM->SetSplittingRule( splittingRule );
    tmpPtrJ->SetSplittingRule( splittingRule );

    this->m_BinaryKNNTreeFixedIntensity  = tmpPtrF;
    this->m_BinaryKNNTreeMovingIntensity = tmpPtrM;
    this->m_BinaryKNNTreeJointIntensity  = tmpPtrJ;

  } // end SetANNkDTree


  /**
	 * ************************ SetANNbdTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNbdTree( unsigned int bucketSize = 2, std::string splittingRule = "ANN_KD_SL_MIDPT",
    std::string shrinkingRule = "ANN_BD_SIMPLE" )
  {
    typename ANNbdTreeType::Pointer tmpPtrF = ANNbdTreeType::New();
    typename ANNbdTreeType::Pointer tmpPtrM = ANNbdTreeType::New();
    typename ANNbdTreeType::Pointer tmpPtrJ = ANNbdTreeType::New();
    
    tmpPtrF->SetBucketSize( bucketSize );
    tmpPtrM->SetBucketSize( bucketSize );
    tmpPtrJ->SetBucketSize( bucketSize );

    tmpPtrF->SetSplittingRule( splittingRule );
    tmpPtrM->SetSplittingRule( splittingRule );
    tmpPtrJ->SetSplittingRule( splittingRule );

    tmpPtrF->SetShrinkingRule( shrinkingRule );
    tmpPtrM->SetShrinkingRule( shrinkingRule );
    tmpPtrJ->SetShrinkingRule( shrinkingRule );

    this->m_BinaryKNNTreeFixedIntensity  = tmpPtrF;
    this->m_BinaryKNNTreeMovingIntensity = tmpPtrM;
    this->m_BinaryKNNTreeJointIntensity  = tmpPtrJ;

  } // end SetANNbdTree


  /**
	 * ************************ SetANNBruteForceTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNBruteForceTree( void )
  {
    this->m_BinaryKNNTreeFixedIntensity  = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeMovingIntensity = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeJointIntensity  = ANNBruteForceTreeType::New();
    
  } // end SetANNBruteForceTree


  /**
	 * ************************ SetANNStandardTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNStandardTreeSearch( unsigned int kNearestNeighbors = 5,
    double errorBound = 0.0 )
  {
    typename ANNStandardTreeSearchType::Pointer tmpPtrF
      = ANNStandardTreeSearchType::New();
    typename ANNStandardTreeSearchType::Pointer tmpPtrM
      = ANNStandardTreeSearchType::New();
    typename ANNStandardTreeSearchType::Pointer tmpPtrJ
      = ANNStandardTreeSearchType::New();

    tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

    tmpPtrF->SetErrorBound( errorBound );
    tmpPtrM->SetErrorBound( errorBound );
    tmpPtrJ->SetErrorBound( errorBound );

    this->m_BinaryKNNTreeSearcherFixedIntensity  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMovingIntensity = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJointIntensity  = tmpPtrJ;

  } // end SetANNStandardTreeSearch


  /**
	 * ************************ SetANNFixedRadiusTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNFixedRadiusTreeSearch( unsigned int kNearestNeighbors = 5,
    double errorBound = 0.0, double squaredRadius = 0.0 )
  {
    typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrF
      = ANNFixedRadiusTreeSearchType::New();
    typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrM
      = ANNFixedRadiusTreeSearchType::New();
    typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrJ
      = ANNFixedRadiusTreeSearchType::New();
    
    tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

    tmpPtrF->SetErrorBound( errorBound );
    tmpPtrM->SetErrorBound( errorBound );
    tmpPtrJ->SetErrorBound( errorBound );

    tmpPtrF->SetSquaredRadius( squaredRadius );
    tmpPtrM->SetSquaredRadius( squaredRadius );
    tmpPtrJ->SetSquaredRadius( squaredRadius );

    this->m_BinaryKNNTreeSearcherFixedIntensity  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMovingIntensity = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJointIntensity  = tmpPtrJ;
    
  } // end SetANNFixedRadiusTreeSearch


  /**
	 * ************************ SetANNPriorityTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNPriorityTreeSearch( unsigned int kNearestNeighbors = 5,
    double errorBound = 0.0 )
  {
    typename ANNPriorityTreeSearchType::Pointer tmpPtrF
      = ANNPriorityTreeSearchType::New();
    typename ANNPriorityTreeSearchType::Pointer tmpPtrM
      = ANNPriorityTreeSearchType::New();
    typename ANNPriorityTreeSearchType::Pointer tmpPtrJ
      = ANNPriorityTreeSearchType::New();
    
    tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
    tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

    tmpPtrF->SetErrorBound( errorBound );
    tmpPtrM->SetErrorBound( errorBound );
    tmpPtrJ->SetErrorBound( errorBound );

    this->m_BinaryKNNTreeSearcherFixedIntensity  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMovingIntensity = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJointIntensity  = tmpPtrJ;

  } // end SetANNPriorityTreeSearch


  /**
	 * ********************* Initialize *****************************
	 */

	template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage> 
		void KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		/** Call the superclass. */
		this->Superclass::Initialize();
		
		/** Cache the number of transformation parameters.*/
		this->m_NumberOfParameters = this->m_Transform->GetNumberOfParameters();

    /** Check if the kNN tree is set. */
    if ( !this->m_BinaryKNNTreeFixedIntensity )
    {
      itkExceptionMacro( << "ERROR: The kNN tree is not set. " );
    }

    /** Check if the kNN tree searcher is set. */
    if ( !this->m_BinaryKNNTreeSearcherFixedIntensity )
    {
      itkExceptionMacro( << "ERROR: The kNN tree searcher is not set. " );
    }
				
	} // end Initialize


  /**
	 * ************************ GetValue *************************
   *
   * Get the match Measure
	 */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    typename KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>::MeasureType
    KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetValue( const TransformParametersType & parameters ) const
  {
    /** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    unsigned long size = sampleContainer->Size();

    /**
     * *************** First create the three list samples ******************
     */

    /** Create list samples. */
    typename ListSampleType::Pointer listSampleFixedIntensity  = ListSampleType::New();
    typename ListSampleType::Pointer listSampleMovingIntensity = ListSampleType::New();
    typename ListSampleType::Pointer listSampleJointIntensity  = ListSampleType::New();

    /** Get the size of the feature vectors. */
    unsigned int fixedSize  = 1 + this->m_NumberOfFixedFeatureImages;
    unsigned int movingSize = 1 + this->m_NumberOfMovingFeatureImages;
    unsigned int jointSize  = fixedSize + movingSize;

    /** Resize them so that enough memory is allocated. */
    listSampleFixedIntensity->SetMeasurementVectorSize( fixedSize );
    listSampleFixedIntensity->Resize( size );
    listSampleMovingIntensity->SetMeasurementVectorSize( movingSize );
    listSampleMovingIntensity->Resize( size );
    listSampleJointIntensity->SetMeasurementVectorSize( jointSize );
    listSampleJointIntensity->Resize( size );

    /** Create variables to store intermediate results. */
    InputPointType  inputPoint;
    OutputPointType transformedPoint;
    std::vector<double> fixedFeatureValues( this->m_NumberOfFixedFeatureImages );
    std::vector<double> movingFeatureValues( this->m_NumberOfMovingFeatureImages );

    /** Loop over the fixed image samples to calculate list samples. */
    for ( unsigned long i = 0; i < size; i++ )
    {
      /** Get the current inputpoint. */
      inputPoint = sampleContainer->GetElement( i ).m_ImageCoordinates;

      /** Transform the inputpoint to get the transformed point. */
      transformedPoint = this->m_Transform->TransformPoint( inputPoint );

      /** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				continue;
			}

      /** In this if-statement the actual addition to the list samples is done. */
      if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
        const RealType & fixedValue = sampleContainer->GetElement( i ).m_ImageValue;
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
       
        /** This is a valid sample: add it to the ListSampleCarrays. */
        listSampleFixedIntensity->SetMeasurement(  this->m_NumberOfPixelsCounted, 0, fixedValue );
        listSampleMovingIntensity->SetMeasurement( this->m_NumberOfPixelsCounted, 0, movingValue );
        listSampleJointIntensity->SetMeasurement(  this->m_NumberOfPixelsCounted, 0, fixedValue );
        listSampleJointIntensity->SetMeasurement(  this->m_NumberOfPixelsCounted,
          1 + this->m_NumberOfFixedFeatureImages, movingValue );

        /** Get and set the values of the fixed feature images. */
        for ( unsigned int j = 0; j < this->m_NumberOfFixedFeatureImages; j++ )
        {
          fixedFeatureValues[ j ] = this->m_FixedFeatureInterpolators[ j ]->Evaluate( inputPoint );
          listSampleFixedIntensity->SetMeasurement(
            this->m_NumberOfPixelsCounted, j + 1, fixedFeatureValues[ j ] );
          listSampleJointIntensity->SetMeasurement(
            this->m_NumberOfPixelsCounted, j + 1, fixedFeatureValues[ j ] );
        }

        /** Get and set the values of the moving feature images. */
        for ( unsigned int j = 0; j < this->m_NumberOfMovingFeatureImages; j++ )
        {
          movingFeatureValues[ j ] = this->m_MovingFeatureInterpolators[ j ]->Evaluate( inputPoint );
          listSampleMovingIntensity->SetMeasurement(
            this->m_NumberOfPixelsCounted, j + 1, movingFeatureValues[ j ] );
          listSampleJointIntensity->SetMeasurement(
            this->m_NumberOfPixelsCounted,
            j + 1 + this->m_NumberOfFixedFeatureImages,
            movingFeatureValues[ j ]
            );
        }

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()
    }

    /** The listSamples are of size sampleContainer->Size(). However, not all of
     * those points made it to the respective list samples. Therefore, we set
     * the actual number of pixels in the sample container, so that the binary
     * trees know where to loop over. This must not be forgotten!
     */
    listSampleFixedIntensity->SetActualSize( this->m_NumberOfPixelsCounted );
    listSampleMovingIntensity->SetActualSize( this->m_NumberOfPixelsCounted );
    listSampleJointIntensity->SetActualSize( this->m_NumberOfPixelsCounted );

    /** Generate the tree for the fixed image samples. */
    this->m_BinaryKNNTreeFixedIntensity->SetSample( listSampleFixedIntensity );
    this->m_BinaryKNNTreeFixedIntensity->GenerateTree();

    /** Generate the tree for the moving image samples. */
    this->m_BinaryKNNTreeMovingIntensity->SetSample( listSampleMovingIntensity );
    this->m_BinaryKNNTreeMovingIntensity->GenerateTree();

    /** Generate the tree for the joint image samples. */
    this->m_BinaryKNNTreeJointIntensity->SetSample( listSampleJointIntensity );
    this->m_BinaryKNNTreeJointIntensity->GenerateTree();

    /**
     * *************** Then estimate the \alpha MI ******************
     *
     * This is done by searching for the nearest neighbours of each point
     * and calculating the distances.
     */

    /** Initialize tree searchers. */
    this->m_BinaryKNNTreeSearcherFixedIntensity
      ->SetBinaryTree( this->m_BinaryKNNTreeFixedIntensity );
    this->m_BinaryKNNTreeSearcherMovingIntensity
      ->SetBinaryTree( this->m_BinaryKNNTreeMovingIntensity );
    this->m_BinaryKNNTreeSearcherJointIntensity
      ->SetBinaryTree( this->m_BinaryKNNTreeJointIntensity );
    
    /** The estimate for the alpha - mutual information is given by:
     *
     *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
     *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
     *
     * where
     *  \alpha is set by the user and refers to \alpha - mutual information
     *  n is the number of samples
     *  k is the number of nearest neighbours
     *  jointLength  is the distances to one of the nearest neighbours in listSampleJointIntensity
     *  fixedLength  is the distances to one of the nearest neighbours in listSampleFixedIntensity
     *  movingLength is the distances to one of the nearest neighbours in listSampleMovingIntensity
     *  \gamma relates to the distance metric and relates to \alpha as: \gamma = ( 1 - \alpha ) * d
     *    where d is the dimension of the feature space (which is in this case 1: intensity only)
     */
 
    /** Temporary variables. */
    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    MeasurementVectorType queryF, queryM, queryJ;
    unsigned int K = this->m_BinaryKNNTreeSearcherFixedIntensity->GetKNearestNeighbors();
    IndexArrayType indicesF, indicesM, indicesJ;
    DistanceArrayType distsF, distsM, distsJ;
    MeasureType enumerator = NumericTraits< MeasureType >::Zero;
    MeasureType denominator = NumericTraits< MeasureType >::Zero;
    AccumulateType contribution = NumericTraits< AccumulateType >::Zero;

    /** Search for the nearest neighbours over all query points, i.e.
     * all (joint) intensities in the listsamples.
     * gamma = d * (1 - alpha ), with d the feature size dimension.
     * In the original paper it is assumed that the mutual information of
     * two feature sets of equal dimension is calculated. If not, don't 
     * take the power to 2 * d * ( 1 - alpha ), but take the power to
     * ( d1 + d2 ) * ( 1 - alpha ), where d1 and d2 are the possibly
     * different dimensions of the two feature sets. So,
     *    gamma = ( d1 + d2 ) * ( 1 - alpha ) / 2.
     */
    double twoGamma = jointSize * ( 1 - this->m_Alpha );
    for ( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
      /** Get the i-th query point. */
      listSampleFixedIntensity->GetMeasurementVector( i, queryF );
      listSampleMovingIntensity->GetMeasurementVector( i, queryM );
      listSampleJointIntensity->GetMeasurementVector( i, queryJ );

      /** Find the K nearest neighbours of the current query point. */
      this->m_BinaryKNNTreeSearcherFixedIntensity->Search( queryF, indicesF, distsF );
      this->m_BinaryKNNTreeSearcherMovingIntensity->Search( queryM, indicesM, distsM );
      this->m_BinaryKNNTreeSearcherJointIntensity->Search( queryJ, indicesJ, distsJ );

      /** Add the distances between the points to get the total graph length. */
      for ( unsigned int j = 0; j < K; j++ )
      {
        enumerator = vcl_sqrt( distsJ[ j ] );
        denominator = vcl_sqrt( vcl_sqrt( distsF[ j ] ) * vcl_sqrt( distsM[ j ] ) );
        if ( denominator > 1e-14 )
        {
          contribution += vcl_pow( enumerator / denominator, twoGamma );
        }
      }
    } // end searching over all query points


    /** Calculate the metric value. */
    MeasureType measure = NumericTraits< AccumulateType >::Zero;
    double number;
    if ( this->m_NumberOfPixelsCounted > 0 && contribution > 1e-14 )
    {
      number = vcl_pow( static_cast<double>( this->m_NumberOfPixelsCounted ), this->m_Alpha );
      measure = vcl_log( contribution / number ) / ( this->m_Alpha - 1.0 );
    }

    /** Return the negative alpha - mutual information. */
    return -measure;

  } // end GetValue


  /**
	 * ************************ GetDerivative *************************
   *
   * Get the Derivative Measure
	 */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {
  } // end GetDerivative


  /**
	 * ************************ GetValueAndDerivative *************************
   *
   * Get both the match Measure and theDerivative Measure
	 */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & value, DerivativeType  & derivative ) const
  {
  } // end GetValueAndDerivative

 
  /**
	 * ************************ PrintSelf *************************
	 */
  
  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
      TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf(os, indent);

    os << indent << "NumberOfParameters: " << this->m_NumberOfParameters << std::endl;
    os << indent << "Alpha: " << this->m_Alpha << std::endl;

    os << indent << "BinaryKNNTreeFixedIntensity: "
      << this->m_BinaryKNNTreeFixedIntensity.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeMovingIntensity: "
      << this->m_BinaryKNNTreeMovingIntensity.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeJointIntensity: "
      << this->m_BinaryKNNTreeJointIntensity.GetPointer() << std::endl;

    os << indent << "BinaryKNNTreeSearcherFixedIntensity: "
      << this->m_BinaryKNNTreeSearcherFixedIntensity.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeSearcherMovingIntensity: "
      << this->m_BinaryKNNTreeSearcherMovingIntensity.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeSearcherJointIntensity: "
      << this->m_BinaryKNNTreeSearcherJointIntensity.GetPointer() << std::endl;

  } // end PrintSelf


} // end namespace itk


#endif // end #ifndef _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx

