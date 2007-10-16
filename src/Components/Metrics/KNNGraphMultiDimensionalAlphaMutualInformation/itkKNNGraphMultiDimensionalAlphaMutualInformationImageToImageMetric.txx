#ifndef _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx
#define _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx

#include "itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric.h"


namespace itk
{

  /**
	 * ************************ Constructor *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric()
  {
    this->SetComputeGradient(false); // don't use the default gradient for now
    this->SetUseImageSampler(true);
    this->m_Alpha = 0.5;

    this->m_BinaryKNNTreeFixed = 0;
    this->m_BinaryKNNTreeMoving = 0;
    this->m_BinaryKNNTreeJoint = 0;
    
    this->m_BinaryKNNTreeSearcherFixed = 0;
    this->m_BinaryKNNTreeSearcherMoving = 0;
    this->m_BinaryKNNTreeSearcherJoint = 0;

  } // end Constructor


  /**
	 * ************************ SetANNkDTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNkDTree( unsigned int bucketSize = 2, std::string splittingRule = "ANN_KD_SL_MIDPT" )
  {
    this->SetANNkDTree( bucketSize, splittingRule, splittingRule, splittingRule );
  } // end SetANNkDTree()


  /**
	 * ************************ SetANNkDTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNkDTree( unsigned int bucketSize, std::string splittingRuleFixed,
    std::string splittingRuleMoving, std::string splittingRuleJoint )
  {
    typename ANNkDTreeType::Pointer tmpPtrF = ANNkDTreeType::New();
    typename ANNkDTreeType::Pointer tmpPtrM = ANNkDTreeType::New();
    typename ANNkDTreeType::Pointer tmpPtrJ = ANNkDTreeType::New();
    
    tmpPtrF->SetBucketSize( bucketSize );
    tmpPtrM->SetBucketSize( bucketSize );
    tmpPtrJ->SetBucketSize( bucketSize );

    tmpPtrF->SetSplittingRule( splittingRuleFixed );
    tmpPtrM->SetSplittingRule( splittingRuleMoving );
    tmpPtrJ->SetSplittingRule( splittingRuleJoint );

    this->m_BinaryKNNTreeFixed  = tmpPtrF;
    this->m_BinaryKNNTreeMoving = tmpPtrM;
    this->m_BinaryKNNTreeJoint  = tmpPtrJ;

  } // end SetANNkDTree()


  /**
	 * ************************ SetANNbdTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNbdTree( unsigned int bucketSize = 2, std::string splittingRule = "ANN_KD_SL_MIDPT",
    std::string shrinkingRule = "ANN_BD_SIMPLE" )
  {
    this->SetANNbdTree( bucketSize, splittingRule, splittingRule, splittingRule,
      shrinkingRule, shrinkingRule, shrinkingRule );
  } // end SetANNbdTree()


  /**
	 * ************************ SetANNbdTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNbdTree( unsigned int bucketSize, std::string splittingRuleFixed,
    std::string splittingRuleMoving, std::string splittingRuleJoint,
    std::string shrinkingRuleFixed, std::string shrinkingRuleMoving,
    std::string shrinkingRuleJoint )
  {
    typename ANNbdTreeType::Pointer tmpPtrF = ANNbdTreeType::New();
    typename ANNbdTreeType::Pointer tmpPtrM = ANNbdTreeType::New();
    typename ANNbdTreeType::Pointer tmpPtrJ = ANNbdTreeType::New();
    
    tmpPtrF->SetBucketSize( bucketSize );
    tmpPtrM->SetBucketSize( bucketSize );
    tmpPtrJ->SetBucketSize( bucketSize );

    tmpPtrF->SetSplittingRule( splittingRuleFixed );
    tmpPtrM->SetSplittingRule( splittingRuleMoving );
    tmpPtrJ->SetSplittingRule( splittingRuleJoint );

    tmpPtrF->SetShrinkingRule( shrinkingRuleFixed );
    tmpPtrM->SetShrinkingRule( shrinkingRuleMoving );
    tmpPtrJ->SetShrinkingRule( shrinkingRuleJoint );

    this->m_BinaryKNNTreeFixed  = tmpPtrF;
    this->m_BinaryKNNTreeMoving = tmpPtrM;
    this->m_BinaryKNNTreeJoint  = tmpPtrJ;

  } // end SetANNbdTree()


  /**
	 * ************************ SetANNBruteForceTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetANNBruteForceTree( void )
  {
    this->m_BinaryKNNTreeFixed  = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeMoving = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeJoint  = ANNBruteForceTreeType::New();
    
  } // end SetANNBruteForceTree()


  /**
	 * ************************ SetANNStandardTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
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

    this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;

  } // end SetANNStandardTreeSearch()


  /**
	 * ************************ SetANNFixedRadiusTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
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

    this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;
    
  } // end SetANNFixedRadiusTreeSearch()


  /**
	 * ************************ SetANNPriorityTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
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

    this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
    this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
    this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;

  } // end SetANNPriorityTreeSearch()


  /**
	 * ********************* Initialize *****************************
	 */

	template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage> 
	void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		/** Call the superclass. */
		this->Superclass::Initialize();
		
	  /** Check if the kNN tree is set. */
    if ( !this->m_BinaryKNNTreeFixed )
    {
      itkExceptionMacro( << "ERROR: The kNN tree is not set. " );
    }

    /** Check if the kNN tree searcher is set. */
    if ( !this->m_BinaryKNNTreeSearcherFixed )
    {
      itkExceptionMacro( << "ERROR: The kNN tree searcher is not set. " );
    }
				
	} // end Initialize()


  /**
	 * ************************ GetValue *************************
   *
   * Get the match Measure
	 */

  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  typename KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>::MeasureType
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetValue( const TransformParametersType & parameters ) const
  {
    /** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

    /**
     * *************** Create the three list samples ******************
     */

    /** Create list samples. */
    ListSamplePointer listSampleFixed  = ListSampleType::New();
    ListSamplePointer listSampleMoving = ListSampleType::New();
    ListSamplePointer listSampleJoint  = ListSampleType::New();

    /** Compute the three list samples. */
    TransformJacobianContainerType dummyJacobians;
    TransformJacobianIndicesContainerType dummyJacobiansIndices;
    SpatialDerivativeContainerType dummySpatialDerivatives;
    this->ComputeListSampleValuesAndDerivativePlusJacobian(
      listSampleFixed, listSampleMoving, listSampleJoint,
      false, dummyJacobians, dummyJacobiansIndices, dummySpatialDerivatives );
  
    /** Check if enough samples were valid. */
    unsigned long size = this->GetImageSampler()->GetOutput()->Size();
    this->CheckNumberOfSamples( size,
      this->m_NumberOfPixelsCounted, this->m_NumberOfPixelsCounted );

    /**
     * *************** Generate the three trees ******************
     *
     * and connect them to the searchers.
     */

    /** Generate the tree for the fixed image samples. */
    this->m_BinaryKNNTreeFixed->SetSample( listSampleFixed );
    this->m_BinaryKNNTreeFixed->GenerateTree();

    /** Generate the tree for the moving image samples. */
    this->m_BinaryKNNTreeMoving->SetSample( listSampleMoving );
    this->m_BinaryKNNTreeMoving->GenerateTree();

    /** Generate the tree for the joint image samples. */
    this->m_BinaryKNNTreeJoint->SetSample( listSampleJoint );
    this->m_BinaryKNNTreeJoint->GenerateTree();

    /** Initialize tree searchers. */
    this->m_BinaryKNNTreeSearcherFixed
      ->SetBinaryTree( this->m_BinaryKNNTreeFixed );
    this->m_BinaryKNNTreeSearcherMoving
      ->SetBinaryTree( this->m_BinaryKNNTreeMoving );
    this->m_BinaryKNNTreeSearcherJoint
      ->SetBinaryTree( this->m_BinaryKNNTreeJoint );

    /**
     * *************** Estimate the \alpha MI ******************
     *
     * This is done by searching for the nearest neighbours of each point
     * and calculating the distances.
     *
     * The estimate for the alpha - mutual information is given by:
     *
     *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
     *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
     *
     * where
     *   - \alpha is set by the user and refers to \alpha - mutual information
     *   - n is the number of samples
     *   - k is the number of nearest neighbours
     *   - jointLength  is the distances to one of the nearest neighbours in listSampleJoint
     *   - fixedLength  is the distances to one of the nearest neighbours in listSampleFixed
     *   - movingLength is the distances to one of the nearest neighbours in listSampleMoving
     *   - \gamma relates to the distance metric and relates to \alpha as:
     *
     *        \gamma = d * ( 1 - \alpha ),
     *
     *     where d is the dimension of the feature space.
     *
     * In the original paper it is assumed that the mutual information of
     * two feature sets of equal dimension is calculated. If not this is not
     * true, then
     *
     *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
     *
     * where d1 and d2 are the possibly different dimensions of the two feature sets.
     */
 
    /** Temporary variables. */
    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    MeasurementVectorType queryF, queryM, queryJ;
    IndexArrayType indicesF, indicesM, indicesJ;
    DistanceArrayType distsF, distsM, distsJ;
    MeasureType enumerator = NumericTraits< MeasureType >::Zero;
    MeasureType denominator = NumericTraits< MeasureType >::Zero;
    AccumulateType contribution = NumericTraits< AccumulateType >::Zero;

    /** Get the size of the feature vectors. */
    unsigned int fixedSize  = 1 + this->m_NumberOfFixedFeatureImages;
    unsigned int movingSize = 1 + this->m_NumberOfMovingFeatureImages;
    unsigned int jointSize  = fixedSize + movingSize;

    /** Get the number of neighbours and \gamma. */
    unsigned int K = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
    double twoGamma = jointSize * ( 1.0 - this->m_Alpha );

    /** Loop over all query points, i.e. all samples. */
    for ( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
      /** Get the i-th query point. */
      listSampleFixed->GetMeasurementVector( i, queryF );
      listSampleMoving->GetMeasurementVector( i, queryM );
      listSampleJoint->GetMeasurementVector( i, queryJ );

      /** Search for the K nearest neighbours of the current query point. */
      this->m_BinaryKNNTreeSearcherFixed->Search( queryF, indicesF, distsF );
      this->m_BinaryKNNTreeSearcherMoving->Search( queryM, indicesM, distsM );
      this->m_BinaryKNNTreeSearcherJoint->Search( queryJ, indicesJ, distsJ );
      
      /** Add the distances between the points to get the total graph length.
       * The outcommented implementation calculates: sum J/sqrt(F*M)
       *
      for ( unsigned int j = 0; j < K; j++ )
      {
        enumerator = vcl_sqrt( distsJ[ j ] );
        denominator = vcl_sqrt( vcl_sqrt( distsF[ j ] ) * vcl_sqrt( distsM[ j ] ) );
        if ( denominator > 1e-14 )
        {
          contribution += vcl_pow( enumerator / denominator, twoGamma );
        }
      }*/

      /** Add the distances of all neighbours of the query point,
       * for the three graphs:
       * sum M / sqrt( sum F * sum M)
       */
      AccumulateType totalDistsF = NumericTraits< AccumulateType >::Zero;
      AccumulateType totalDistsM = NumericTraits< AccumulateType >::Zero;
      AccumulateType totalDistsJ = NumericTraits< AccumulateType >::Zero;
      for ( unsigned int j = 0; j < K; j++ )
      {
        totalDistsJ += vcl_sqrt( distsJ[ j ] );
        totalDistsF += vcl_sqrt( distsF[ j ] );
        totalDistsM += vcl_sqrt( distsM[ j ] );
      } // end loop over the K neighbours
      
      /** Calculate the contribution of this query point. */
      denominator = vcl_sqrt( totalDistsF * totalDistsM );
      if ( denominator > 1e-14 )
      {
        contribution += vcl_pow( totalDistsJ / denominator, twoGamma );
      }
    } // end searching over all query points

    /**
     * *************** Finally, calculate the metric value \alpha MI ******************
     */

    MeasureType measure = NumericTraits< AccumulateType >::Zero;
    double n, number;
    if ( contribution > 1e-14 )
    {
      n = static_cast<double>( this->m_NumberOfPixelsCounted );
      number = vcl_pow( n, this->m_Alpha );
      measure = vcl_log( contribution / number ) / ( this->m_Alpha - 1.0 );
    }

    /** Return the negative alpha - mutual information. */
    return -measure;

  } // end GetValue()


  /**
	 * ************************ GetDerivative *************************
   *
   * Get the Derivative Measure
	 */

  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {
    /** When the derivative is calculated, all information for calculating
		 * the metric value is available. It does not cost anything to calculate
		 * the metric value now. Therefore, we have chosen to only implement the
		 * GetValueAndDerivative(), supplying it with a dummy value variable. */
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
    this->GetValueAndDerivative( parameters, dummyvalue, derivative );

  } // end GetDerivative()


  /**
	 * ************************ GetValueAndDerivative *************************
   *
   * Get both the match Measure and theDerivative Measure
	 */

  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const
  {
    /** Initialize some variables. */
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->m_NumberOfParameters );
		derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

    /**
     * *************** Create the three list samples ******************
     */

    /** Create list samples. */
    ListSamplePointer listSampleFixed  = ListSampleType::New();
    ListSamplePointer listSampleMoving = ListSampleType::New();
    ListSamplePointer listSampleJoint  = ListSampleType::New();

    /** Compute the three list samples and the derivatives. */
    TransformJacobianContainerType jacobianContainer;
    TransformJacobianIndicesContainerType jacobianIndicesContainer;
    SpatialDerivativeContainerType spatialDerivativesContainer;
    this->ComputeListSampleValuesAndDerivativePlusJacobian(
      listSampleFixed, listSampleMoving, listSampleJoint,
      true, jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer );
  
    /** Check if enough samples were valid. */
    unsigned long size = this->GetImageSampler()->GetOutput()->Size();
    this->CheckNumberOfSamples( size,
      this->m_NumberOfPixelsCounted, this->m_NumberOfPixelsCounted );

    /**
     * *************** Generate the three trees ******************
     *
     * and connect them to the searchers.
     */

    /** Generate the tree for the fixed image samples. */
    this->m_BinaryKNNTreeFixed->SetSample( listSampleFixed );
    this->m_BinaryKNNTreeFixed->GenerateTree();

    /** Generate the tree for the moving image samples. */
    this->m_BinaryKNNTreeMoving->SetSample( listSampleMoving );
    this->m_BinaryKNNTreeMoving->GenerateTree();

    /** Generate the tree for the joint image samples. */
    this->m_BinaryKNNTreeJoint->SetSample( listSampleJoint );
    this->m_BinaryKNNTreeJoint->GenerateTree();

    /** Initialize tree searchers. */
    this->m_BinaryKNNTreeSearcherFixed
      ->SetBinaryTree( this->m_BinaryKNNTreeFixed );
    this->m_BinaryKNNTreeSearcherMoving
      ->SetBinaryTree( this->m_BinaryKNNTreeMoving );
    this->m_BinaryKNNTreeSearcherJoint
      ->SetBinaryTree( this->m_BinaryKNNTreeJoint );

    /**
     * *************** Estimate the \alpha MI and its derivatives ******************
     *
     * This is done by searching for the nearest neighbours of each point
     * and calculating the distances.
     *
     * The estimate for the alpha - mutual information is given by:
     *
     *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
     *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
     *
     * where
     *   - \alpha is set by the user and refers to \alpha - mutual information
     *   - n is the number of samples
     *   - k is the number of nearest neighbours
     *   - jointLength  is the distances to one of the nearest neighbours in listSampleJoint
     *   - fixedLength  is the distances to one of the nearest neighbours in listSampleFixed
     *   - movingLength is the distances to one of the nearest neighbours in listSampleMoving
     *   - \gamma relates to the distance metric and relates to \alpha as:
     *
     *        \gamma = d * ( 1 - \alpha ),
     *
     *     where d is the dimension of the feature space.
     *
     * In the original paper it is assumed that the mutual information of
     * two feature sets of equal dimension is calculated. If not this is not
     * true, then
     *
     *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
     *
     * where d1 and d2 are the possibly different dimensions of the two feature sets.
     */
 
    /** Temporary variables. */
    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    MeasurementVectorType z_F, z_M, z_J, z_M_ip, z_J_ip, diff_M, diff_J;
    IndexArrayType indices_F, indices_M, indices_J;
    DistanceArrayType distances_F, distances_M, distances_J;
    MeasureType distance_F, distance_M, distance_J;

    MeasureType H, G, Gpow;
    AccumulateType sumG = NumericTraits< AccumulateType >::Zero;

    DerivativeType contribution( this->GetNumberOfParameters() );
    contribution.Fill( NumericTraits< DerivativeValueType >::Zero );
    DerivativeType dGamma_M( this->GetNumberOfParameters() );
    DerivativeType dGamma_J( this->GetNumberOfParameters() );

    /** Get the size of the feature vectors. */
    unsigned int fixedSize  = 1 + this->m_NumberOfFixedFeatureImages;
    unsigned int movingSize = 1 + this->m_NumberOfMovingFeatureImages;
    unsigned int jointSize  = fixedSize + movingSize;

    /** Get the number of neighbours and \gamma. */
    unsigned int k = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
    double twoGamma = jointSize * ( 1.0 - this->m_Alpha );

    /** Loop over all query points, i.e. all samples. */
    for ( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
      /** Get the i-th query point. */
      listSampleFixed->GetMeasurementVector(  i, z_F );
      listSampleMoving->GetMeasurementVector( i, z_M );
      listSampleJoint->GetMeasurementVector(  i, z_J );

      /** Search for the K nearest neighbours of the current query point. */
      this->m_BinaryKNNTreeSearcherFixed->Search(  z_F, indices_F, distances_F );
      this->m_BinaryKNNTreeSearcherMoving->Search( z_M, indices_M, distances_M );
      this->m_BinaryKNNTreeSearcherJoint->Search(  z_J, indices_J, distances_J );
   
      /** Variables to compute the measure and its derivative. */
      AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

      SpatialDerivativeType D1sparse, D2sparse_M, D2sparse_J;
      D1sparse = spatialDerivativesContainer[ i ] * jacobianContainer[ i ];
      SpatialDerivativeType Dfull_M( movingSize, this->m_NumberOfParameters );
      SpatialDerivativeType Dfull_J( movingSize, this->m_NumberOfParameters );
      
      dGamma_M.Fill( NumericTraits< DerivativeValueType >::Zero );
      dGamma_J.Fill( NumericTraits< DerivativeValueType >::Zero );

      /** Loop over the neighbours. */
      for ( unsigned int p = 0; p < k; p++ )
      {
        /** Get the neighbour point z_ip^M and the difference with z_i^M. */
        listSampleMoving->GetMeasurementVector( indices_M[ p ], z_M_ip );
        listSampleMoving->GetMeasurementVector( indices_J[ p ], z_J_ip );
        diff_M = z_M - z_M_ip;
        diff_J = z_M - z_J_ip;

        /** Get the distances. */
        distance_F = vcl_sqrt( distances_F[ p ] );
        distance_M = vcl_sqrt( distances_M[ p ] );
        distance_J = vcl_sqrt( distances_J[ p ] );

        /** Compute Gamma's. */
        Gamma_F += distance_F;
        Gamma_M += distance_M;
        Gamma_J += distance_J;

        /** Compute derivatives. */
        D2sparse_M = spatialDerivativesContainer[ indices_M[ p ] ]
          * jacobianContainer[ indices_M[ p ] ];
        D2sparse_J = spatialDerivativesContainer[ indices_J[ p ] ]
          * jacobianContainer[ indices_J[ p ] ];
      
        this->ComputeImageJacobianDifference(
          D1sparse, D2sparse_M, D2sparse_J,
          jacobianIndicesContainer[ i ],
          jacobianIndicesContainer[ indices_M[ p ] ],
          jacobianIndicesContainer[ indices_J[ p ] ],
          Dfull_M, Dfull_J );
        diff_M.post_multiply( Dfull_M );
        diff_J.post_multiply( Dfull_J );
        dGamma_M += diff_M / distance_M;
        dGamma_J += diff_J / distance_J;

      } // end loop over the K neighbours
      
      /** Compute contributions. */
      H = vcl_sqrt( Gamma_F * Gamma_M );
      if ( H > 1e-14 )
      {
        /** Compute some sums. */
        G = Gamma_J / H;
        sumG += vcl_pow( G, twoGamma );
        
        /** Compute the contribution to the derivative. */
        Gpow = vcl_pow( G, twoGamma - 1.0 );
        contribution += ( Gpow / H ) * ( dGamma_J - ( 0.5 * Gamma_J / Gamma_M ) * dGamma_M );
      }
      
    } // end looping over all query points

    /**
     * *************** Finally, calculate the metric value and derivative ******************
     */

    /** Compute the value. */
    double n, number;
    if ( sumG > 1e-14 )
    {
      n = static_cast<double>( this->m_NumberOfPixelsCounted );
      number = vcl_pow( n, this->m_Alpha );
      measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );
    }
    value = -measure;

    /** Compute the derivative (-2.0 * d = -jointSize). */
    derivative = ( -static_cast<AccumulateType>( jointSize ) / sumG ) * contribution;
  
  } // end GetValueAndDerivative()

 
  /**
	 * ************************ ComputeListSampleValuesAndDerivativePlusJacobian *************************
   *
   * 
	 */

  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::ComputeListSampleValuesAndDerivativePlusJacobian(
      const ListSamplePointer & listSampleFixed,
      const ListSamplePointer & listSampleMoving,
      const ListSamplePointer & listSampleJoint,
      const bool & doDerivative,
      TransformJacobianContainerType & jacobianContainer,
      TransformJacobianIndicesContainerType & jacobianIndicesContainer,
      SpatialDerivativeContainerType & spatialDerivativesContainer ) const
  {
    /** Initialize. */
    this->m_NumberOfPixelsCounted = 0;
    jacobianContainer.resize( 0 );
    jacobianIndicesContainer.resize( 0 );
    spatialDerivativesContainer.resize( 0 );

		/** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    unsigned long size = sampleContainer->Size();

    /** Create an iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Get the size of the feature vectors. */
    unsigned int fixedSize  = 1 + this->m_NumberOfFixedFeatureImages;
    unsigned int movingSize = 1 + this->m_NumberOfMovingFeatureImages;
    unsigned int jointSize  = fixedSize + movingSize;

    /** Resize the list samples so that enough memory is allocated. */
    listSampleFixed->SetMeasurementVectorSize( fixedSize );
    listSampleFixed->Resize( size );
    listSampleMoving->SetMeasurementVectorSize( movingSize );
    listSampleMoving->Resize( size );
    listSampleJoint->SetMeasurementVectorSize( jointSize );
    listSampleJoint->Resize( size );

    /** Create variables to store intermediate results. */
    RealType movingImageValue, movingMaskValue;
    MovingImagePointType mappedPoint;
    double fixedFeatureValue = 0.0;
    double movingFeatureValue = 0.0;

    /** Loop over the fixed image samples to calculate the list samples. */
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
      /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;

      /** Transform point and check if it is inside the bspline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside moving mask. */
      if ( sampleOk )
      {
        this->EvaluateMovingMaskValueAndDerivative( mappedPoint, movingMaskValue, 0 );
        sampleOk = movingMaskValue > 1e-10;
      }

      /** Compute the moving image value M(T(x)) and possibly the
       * derivative dM/dx and check if the point is inside the
       * moving image buffer. */
      MovingImageDerivativeType movingImageDerivative;
      if ( sampleOk )
      {
        if ( doDerivative )
        {
          sampleOk = this->EvaluateMovingImageValueAndDerivative(
            mappedPoint, movingImageValue, &movingImageDerivative );
        }
        else
        {
          sampleOk = this->EvaluateMovingImageValueAndDerivative(
            mappedPoint, movingImageValue, 0 );
        }
      }

      /** This is a valid sample: in this if-statement the actual
       * addition to the list samples is done.
       */
      if ( sampleOk )
			{
        /** Get the fixed image value. */
        const RealType & fixedImageValue = static_cast<RealType>(
          (*fiter).Value().m_ImageValue );

        /** Add the samples to the ListSampleCarrays. */
        listSampleFixed->SetMeasurement(  this->m_NumberOfPixelsCounted, 0, fixedImageValue );
        listSampleMoving->SetMeasurement( this->m_NumberOfPixelsCounted, 0, movingImageValue );
        listSampleJoint->SetMeasurement(  this->m_NumberOfPixelsCounted, 0, fixedImageValue );
        listSampleJoint->SetMeasurement(  this->m_NumberOfPixelsCounted,
          1 + this->m_NumberOfFixedFeatureImages, movingImageValue );

        /** Get and set the values of the fixed feature images. */
        for ( unsigned int j = 0; j < this->m_NumberOfFixedFeatureImages; j++ )
        {
          fixedFeatureValue = this->m_FixedFeatureInterpolators[ j ]->Evaluate( fixedPoint );
          listSampleFixed->SetMeasurement(
            this->m_NumberOfPixelsCounted, j + 1, fixedFeatureValue );
          listSampleJoint->SetMeasurement(
            this->m_NumberOfPixelsCounted, j + 1, fixedFeatureValue );
        }

        /** Get and set the values of the moving feature images. */
        for ( unsigned int j = 0; j < this->m_NumberOfMovingFeatureImages; j++ )
        {
          movingFeatureValue = this->m_MovingFeatureInterpolators[ j ]->Evaluate( fixedPoint );
          listSampleMoving->SetMeasurement(
            this->m_NumberOfPixelsCounted,
            j + 1,
            movingFeatureValue );
          listSampleJoint->SetMeasurement(
            this->m_NumberOfPixelsCounted,
            j + 2 + this->m_NumberOfFixedFeatureImages,
            movingFeatureValue );
        }

        /** Compute additional stuff for the computation of the derivative, if necessary.
         * - the Jacobian of the transform: dT/dmu(x_i).
         * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
         */
        if ( doDerivative )
        {
          /** Get the TransformJacobian dT/dmu. */
          jacobianContainer.push_back( this->EvaluateTransformJacobian( fixedPoint ) );
          jacobianIndicesContainer.push_back( this->m_NonZeroJacobianIndices );

          /** Get the spatial derivative of the moving image. */
          SpatialDerivativeType spatialDerivatives(
            1 + this->m_NumberOfMovingFeatureImages,
            this->FixedImageDimension );
          spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

          /** Get the spatial derivatives of the moving feature images. */
          SpatialDerivativeType movingFeatureImageDerivatives(
            this->m_NumberOfMovingFeatureImages,
            this->FixedImageDimension );
          this->EvaluateMovingFeatureImageDerivatives(
            mappedPoint, movingFeatureImageDerivatives );
          spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

          /** Put the spatial derivatives of this sample into the container. */
          spatialDerivativesContainer.push_back( spatialDerivatives );

        } // end if doDerivative
       
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if sampleOk

    } // end for loop over the image sample container

    /** The listSamples are of size sampleContainer->Size(). However, not all of
     * those points made it to the respective list samples. Therefore, we set
     * the actual number of pixels in the sample container, so that the binary
     * trees know where to loop over. This must not be forgotten!
     */
    listSampleFixed->SetActualSize( this->m_NumberOfPixelsCounted );
    listSampleMoving->SetActualSize( this->m_NumberOfPixelsCounted );
    listSampleJoint->SetActualSize( this->m_NumberOfPixelsCounted );

  } // end ComputeListSampleValuesAndDerivativePlusJacobian()


  /**
	 * ************************ EvaluateMovingFeatureImageDerivatives *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::EvaluateMovingFeatureImageDerivatives(
    const MovingImagePointType & mappedPoint,
    SpatialDerivativeType & featureGradients ) const
  {
    /** Convert point to a continous index. */
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinousIndex( mappedPoint, cindex );

    /** Compute the spatial derivative for all feature images:
     * - either by calling a special function that only B-spline
     *   interpolators have,
     * - or by using a finite difference approximation of the
     *   pre-computed gradient images.
     * \todo: for now we only implement the first option.
     */
    if ( this->m_FeatureInterpolatorsAreBSpline && !this->GetComputeGradient() )
    {
      /** Computed moving image gradient using derivative BSpline kernel. */
      MovingImageDerivativeType gradient;
      for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; ++i )
      {
        /** Compute the gradient at feature image i. */
        gradient = this
          ->m_MovingFeatureBSplineInterpolators[ i ]
          ->EvaluateDerivativeAtContinuousIndex( cindex );

        /** Set the gradient into the Array2D. */
        featureGradients.set_row( i, gradient.GetDataPointer() );
      } // end for-loop
    } // end if
    /*
    else
    {
      /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
      * It is assumed that the gradient image is computed beforehand.
      *

      /** Round the continuous index to the nearest neighbour. *
      MovingImageIndexType index;
      for ( unsigned int j = 0; j < MovingImageDimension; j++ )
      {
        index[ j ] = static_cast<long>( vnl_math_rnd( cindex[ j ] ) );
      }

      MovingImageDerivativeType gradient;
      for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; ++i )
      {
        /** Compute the gradient at feature image i. *
        gradient = this->m_GradientFeatureImage[ i ]->GetPixel( index );

        /** Set the gradient into the Array2D. *
        featureGradients.set_column( i, gradient.GetDataPointer() );
      } // end for-loop
    } // end if */

  } // end EvaluateMovingFeatureImageDerivatives()


  /**
	 * ************************ ComputeImageJacobianDifference *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::ComputeImageJacobianDifference(
    SpatialDerivativeType & D1sparse,
    SpatialDerivativeType & D2sparse_M,
    SpatialDerivativeType & D2sparse_J,
    ParameterIndexArrayType & D1indices,
    ParameterIndexArrayType & D2indices_M,
    ParameterIndexArrayType & D2indices_J,
    SpatialDerivativeType & Dfull_M,
    SpatialDerivativeType & Dfull_J ) const
  {
    /** Set Dfull_M = Dfull_J = D1sparse. */
    Dfull_M.Fill( NumericTraits<DerivativeValueType>::Zero );
    for ( unsigned int i = 0; i < D1indices.GetSize(); ++i )
    {
      Dfull_M.set_column( D1indices[ i ], D1sparse.get_column( i ) );
    }
    Dfull_J = Dfull_M;
    
    /** Subtract D2sparse_M from Dfull_M. */
    for ( unsigned int i = 0; i < D2indices_M.GetSize(); ++i )
    {
      Dfull_M.set_column( D2indices_M[ i ],
        Dfull_M.get_column( D2indices_M[ i ] ) - D2sparse_M.get_column( i ) );
    }

    /** Subtract D2sparse_J from Dfull_J. */
    for ( unsigned int i = 0; i < D2indices_J.GetSize(); ++i )
    {
      Dfull_J.set_column( D2indices_J[ i ],
        Dfull_J.get_column( D2indices_J[ i ] ) - D2sparse_J.get_column( i ) );
    }

  } // end ComputeImageJacobianDifference()


  /**
	 * ************************ PrintSelf *************************
	 */
  
  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric<
    TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "NumberOfParameters: " << this->m_NumberOfParameters << std::endl;
    os << indent << "Alpha: " << this->m_Alpha << std::endl;

    os << indent << "BinaryKNNTreeFixed: "
      << this->m_BinaryKNNTreeFixed.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeMoving: "
      << this->m_BinaryKNNTreeMoving.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeJoint: "
      << this->m_BinaryKNNTreeJoint.GetPointer() << std::endl;

    os << indent << "BinaryKNNTreeSearcherFixed: "
      << this->m_BinaryKNNTreeSearcherFixed.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeSearcherMoving: "
      << this->m_BinaryKNNTreeSearcherMoving.GetPointer() << std::endl;
    os << indent << "BinaryKNNTreeSearcherJoint: "
      << this->m_BinaryKNNTreeSearcherJoint.GetPointer() << std::endl;

  } // end PrintSelf()


} // end namespace itk


#endif // end #ifndef _itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_txx

