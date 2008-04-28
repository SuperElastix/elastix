/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkKNNGraphAlphaMutualInformationImageToImageMetric_txx
#define _itkKNNGraphAlphaMutualInformationImageToImageMetric_txx

#include "itkKNNGraphAlphaMutualInformationImageToImageMetric.h"

#include "elxTimer.h"


namespace itk
{

  /**
	 * ************************ Constructor *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::KNNGraphAlphaMutualInformationImageToImageMetric()
  {
    this->SetComputeGradient( false ); // don't use the default gradient
    this->SetUseImageSampler( true );
    this->m_Alpha = 0.99;
    this->m_AvoidDivisionBy = 1e-10;

    this->m_BinaryKNNTreeFixed = 0;
    this->m_BinaryKNNTreeMoving = 0;
    this->m_BinaryKNNTreeJoint = 0;
    
    this->m_BinaryKNNTreeSearcherFixed = 0;
    this->m_BinaryKNNTreeSearcherMoving = 0;
    this->m_BinaryKNNTreeSearcherJoint = 0;

m_UseSlow = false;

  } // end Constructor()


  /**
	 * ************************ SetANNkDTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNkDTree( unsigned int bucketSize, std::string splittingRule )
  {
    this->SetANNkDTree( bucketSize, splittingRule, splittingRule, splittingRule );

  } // end SetANNkDTree()


  /**
	 * ************************ SetANNkDTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNbdTree( unsigned int bucketSize,
      std::string splittingRule,
      std::string shrinkingRule )
  {
    this->SetANNbdTree( bucketSize, splittingRule, splittingRule, splittingRule,
      shrinkingRule, shrinkingRule, shrinkingRule );

  } // end SetANNbdTree()


  /**
	 * ************************ SetANNbdTree *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNBruteForceTree( void )
  {
    this->m_BinaryKNNTreeFixed  = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeMoving = ANNBruteForceTreeType::New();
    this->m_BinaryKNNTreeJoint  = ANNBruteForceTreeType::New();
    
  } // end SetANNBruteForceTree()


  /**
	 * ************************ SetANNStandardTreeSearch *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNStandardTreeSearch(
      unsigned int kNearestNeighbors,
      double errorBound )
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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNFixedRadiusTreeSearch(
      unsigned int kNearestNeighbors,
      double errorBound,
      double squaredRadius )
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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetANNPriorityTreeSearch(
      unsigned int kNearestNeighbors,
      double errorBound )
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

	template <class TFixedImage, class TMovingImage>
	void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
		::Initialize( void ) throw ( ExceptionObject )
	{
		/** Call the superclass. */
		this->Superclass::Initialize();
		
	  /** Check if the kNN trees are set. We only need to check the fixed tree. */
    if ( !this->m_BinaryKNNTreeFixed )
    {
      itkExceptionMacro( << "ERROR: The kNN tree is not set. " );
    }

    /** Check if the kNN tree searchers are set. We only need to check the fixed searcher. */
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

  template <class TFixedImage, class TMovingImage>
  typename KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
  ::GetValue( const TransformParametersType & parameters ) const
  {
    /** Initialize some variables. */
    MeasureType measure = NumericTraits< MeasureType >::Zero;

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
    TransformJacobianContainerType dummyJacobianContainer;
    TransformJacobianIndicesContainerType dummyJacobianIndicesContainer;
    SpatialDerivativeContainerType dummySpatialDerivativesContainer;
    this->ComputeListSampleValuesAndDerivativePlusJacobian(
      listSampleFixed, listSampleMoving, listSampleJoint,
      false, dummyJacobianContainer, dummyJacobianIndicesContainer,
      dummySpatialDerivativesContainer );
  
    /** Check if enough samples were valid. */
    unsigned long size = this->GetImageSampler()->GetOutput()->Size();
    this->CheckNumberOfSamples( size,
      this->m_NumberOfPixelsCounted );

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
    MeasurementVectorType z_F, z_M, z_J;
    IndexArrayType indices_F, indices_M, indices_J;
    DistanceArrayType distances_F, distances_M, distances_J;

    MeasureType H, G;
    AccumulateType sumG = NumericTraits< AccumulateType >::Zero;
    
    /** Get the size of the feature vectors. */
    unsigned int fixedSize  = this->GetNumberOfFixedImages();
    unsigned int movingSize = this->GetNumberOfMovingImages();
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

      /** Variables to compute the measure. */
      AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

      /** Loop over the neighbours. */
      for ( unsigned int p = 0; p < k; p++ )
      {
        Gamma_F += vcl_sqrt( distances_F[ p ] );
        Gamma_M += vcl_sqrt( distances_M[ p ] );
        Gamma_J += vcl_sqrt( distances_J[ p ] );
      } // end loop over the k neighbours
      
      /** Calculate the contribution of this query point. */
      H = vcl_sqrt( Gamma_F * Gamma_M );
      if ( H > this->m_AvoidDivisionBy )
      {
        /** Compute some sums. */
        G = Gamma_J / H;
        sumG += vcl_pow( G, twoGamma );
      }
    } // end looping over all query points

    /**
     * *************** Finally, calculate the metric value \alpha MI ******************
     */

    double n, number;
    if ( sumG > this->m_AvoidDivisionBy )
    {
      /** Compute the measure. */
      n = static_cast<double>( this->m_NumberOfPixelsCounted );
      number = vcl_pow( n, this->m_Alpha );
      measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );
    }

    /** Return the negative alpha - mutual information. */
    return -measure;

  } // end GetValue()


  /**
	 * ************************ GetDerivative *************************
   *
   * Get the Derivative Measure
	 */

  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
  ::GetDerivative(
    const TransformParametersType & parameters,
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

  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
  ::GetValueAndDerivative(
    const TransformParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const
  {
    //
    typename tmr::Timer::Pointer timer1 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer2 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer3 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer4 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer5 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer6 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer7 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer8 = tmr::Timer::New();
    typename tmr::Timer::Pointer timer9 = tmr::Timer::New();

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
    timer1->StartTimer();
    TransformJacobianContainerType jacobianContainer;
    TransformJacobianIndicesContainerType jacobianIndicesContainer;
    SpatialDerivativeContainerType spatialDerivativesContainer;
    this->ComputeListSampleValuesAndDerivativePlusJacobian(
      listSampleFixed, listSampleMoving, listSampleJoint,
      true, jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer );
    timer1->StopTimer();
  
    /** Check if enough samples were valid. */
    unsigned long size = this->GetImageSampler()->GetOutput()->Size();
    this->CheckNumberOfSamples( size, this->m_NumberOfPixelsCounted );

    /**
     * *************** Generate the three trees ******************
     *
     * and connect them to the searchers.
     */

    timer2->StartTimer();

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

    timer2->StopTimer();

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
    unsigned int fixedSize  = this->GetNumberOfFixedImages();
    unsigned int movingSize = this->GetNumberOfMovingImages();
    unsigned int jointSize  = fixedSize + movingSize;

    /** Get the number of neighbours and \gamma. */
    unsigned int k = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
    double twoGamma = jointSize * ( 1.0 - this->m_Alpha );

    /** Loop over all query points, i.e. all samples. */
    double time3 = 0.0;
    double time4 = 0.0;
    double time5 = 0.0;
    double time6 = 0.0;
    double time7 = 0.0;
    double time8 = 0.0;
    double time9 = 0.0;
    for ( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
      /** Get the i-th query point. */
      listSampleFixed->GetMeasurementVector(  i, z_F );
      listSampleMoving->GetMeasurementVector( i, z_M );
      listSampleJoint->GetMeasurementVector(  i, z_J );

      /** Search for the K nearest neighbours of the current query point. */
      timer3->StartTimer();
      this->m_BinaryKNNTreeSearcherFixed->Search(  z_F, indices_F, distances_F );
      this->m_BinaryKNNTreeSearcherMoving->Search( z_M, indices_M, distances_M );
      this->m_BinaryKNNTreeSearcherJoint->Search(  z_J, indices_J, distances_J );
      timer3->StopTimer();
      time3 += timer3->GetElapsedClockSec();
   
      /** Variables to compute the measure and its derivative. */
      timer4->StartTimer();
      AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
      AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

      SpatialDerivativeType D1sparse, D2sparse_M, D2sparse_J;
      D1sparse = spatialDerivativesContainer[ i ] * jacobianContainer[ i ];
      SpatialDerivativeType Dfull_M( movingSize, this->m_NumberOfParameters );
      SpatialDerivativeType Dfull_J( movingSize, this->m_NumberOfParameters );
      
      dGamma_M.Fill( NumericTraits< DerivativeValueType >::Zero );
      dGamma_J.Fill( NumericTraits< DerivativeValueType >::Zero );
      timer4->StopTimer();
      time4 += timer4->GetElapsedClockSec();

      /** Loop over the neighbours. */
      timer5->StartTimer();
      for ( unsigned int p = 0; p < k; p++ )
      {
        timer9->StartTimer();
        /** Get the neighbour point z_ip^M. */
        listSampleMoving->GetMeasurementVector( indices_M[ p ], z_M_ip );
        listSampleMoving->GetMeasurementVector( indices_J[ p ], z_J_ip );
        
        /** Get the distances. */
        distance_F = vcl_sqrt( distances_F[ p ] );
        distance_M = vcl_sqrt( distances_M[ p ] );
        distance_J = vcl_sqrt( distances_J[ p ] );

        /** Compute Gamma's. */
        Gamma_F += distance_F;
        Gamma_M += distance_M;
        Gamma_J += distance_J;
        
        /** Get the difference of z_ip^M with z_i^M. */
        diff_M = z_M - z_M_ip;
        diff_J = z_M - z_J_ip;

        /** Compute derivatives. */
        D2sparse_M = spatialDerivativesContainer[ indices_M[ p ] ]
          * jacobianContainer[ indices_M[ p ] ];
        D2sparse_J = spatialDerivativesContainer[ indices_J[ p ] ]
          * jacobianContainer[ indices_J[ p ] ];

        timer9->StopTimer();
        time9 += timer9->GetElapsedClockSec();
      
        if ( this->m_UseSlow )
        {
        /** Compute ( D1sparse - D2sparse_M ) and ( D1sparse - D2sparse_J ).
         * The function returns the full matrices.
         */
        timer7->StartTimer();
        this->ComputeImageJacobianDifference(
          D1sparse, D2sparse_M, D2sparse_J,
          jacobianIndicesContainer[ i ],
          jacobianIndicesContainer[ indices_M[ p ] ],
          jacobianIndicesContainer[ indices_J[ p ] ],
          Dfull_M, Dfull_J );
        timer7->StopTimer();
        time7 += timer7->GetElapsedClockSec();

        timer8->StartTimer();
        diff_M.post_multiply( Dfull_M );
        diff_J.post_multiply( Dfull_J );

        /** Only compute stuff if all distances are large enough. */
        if ( distance_M > this->m_AvoidDivisionBy )
        {
          dGamma_M += diff_M / distance_M;
        }
        if ( distance_J > this->m_AvoidDivisionBy )
        {
          dGamma_J += diff_J / distance_J;
        }
        timer8->StopTimer();
        time8 += timer8->GetElapsedClockSec();
        }
        else
        {
          timer7->StartTimer();

          /** Only compute stuff if all distances are large enough. */
          bool doupdate = false;
          if ( distance_M > this->m_AvoidDivisionBy )
          {
            diff_M /= distance_M;
            doupdate = true;
          }
          if ( distance_J > this->m_AvoidDivisionBy )
          {
            diff_J /= distance_J;
            doupdate &= true;
          }
          if ( doupdate )
          {
            this->ComputeImageJacobianDifference2(
              D1sparse, D2sparse_M, D2sparse_J,
              jacobianIndicesContainer[ i ],
              jacobianIndicesContainer[ indices_M[ p ] ],
              jacobianIndicesContainer[ indices_J[ p ] ],
              diff_M, diff_J,
              distance_M, distance_J,
              dGamma_M, dGamma_J );
          }
            timer7->StopTimer();
          time7 += timer7->GetElapsedClockSec();
        }

      } // end loop over the k neighbours
      timer5->StopTimer();
      time5 += timer5->GetElapsedClockSec();
      
      /** Compute contributions. */
      timer6->StartTimer();
      H = vcl_sqrt( Gamma_F * Gamma_M );
      if ( H > this->m_AvoidDivisionBy )
      {
        /** Compute some sums. */
        G = Gamma_J / H;
        sumG += vcl_pow( G, twoGamma );
        
        /** Compute the contribution to the derivative. */
        Gpow = vcl_pow( G, twoGamma - 1.0 );
        contribution += ( Gpow / H ) * ( dGamma_J - ( 0.5 * Gamma_J / Gamma_M ) * dGamma_M );
      }
      timer6->StopTimer();
      time6 += timer6->GetElapsedClockSec();
     
    } // end looping over all query points

    /**
     * *************** Finally, calculate the metric value and derivative ******************
     */

    /** Compute the value. */
    double n, number;
    if ( sumG > this->m_AvoidDivisionBy )
    {
      /** Compute the measure. */
      n = static_cast<double>( this->m_NumberOfPixelsCounted );
      number = vcl_pow( n, this->m_Alpha );
      measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );

      /** Compute the derivative (-2.0 * d = -jointSize). */
      derivative = ( static_cast<AccumulateType>( jointSize ) / sumG ) * contribution;
    }
    value = -measure;

    /** Print times *
    std::cout << std::endl;
    std::cout << "ComputeListSampleValuesAndDerivativePlusJacobian:\n\t"
      << timer1->PrintElapsedClockSec() << std::endl;
    std::cout << "Setting up kD trees:\n\t"
      << timer2->PrintElapsedClockSec() << std::endl;
    std::cout << "Searching kD trees:\n\t"
      << time3 << std::endl;
    std::cout << "Compute D1sparse:\n\t"
      << time4 << std::endl;
    std::cout << "Loop over k neighbours:\n\t"
      << time5 << std::endl;
    std::cout << "Update contributions:\n\t"
      << time6 << std::endl;
    std::cout << "ComputeImageJacobianDifference:\n\t"
      << time7 << std::endl;
    std::cout << "Compute dGamma_M and dGamma_J:\n\t"
      << time8 << std::endl;
    std::cout << "Compute sparse derivatives:\n\t"
      << time9 << std::endl; */
    
  
  } // end GetValueAndDerivative()

 
  /**
	 * ************************ ComputeListSampleValuesAndDerivativePlusJacobian *************************
	 */

  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
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
    unsigned int fixedSize  = this->GetNumberOfFixedImages();
    unsigned int movingSize = this->GetNumberOfMovingImages();
    unsigned int jointSize  = fixedSize + movingSize;

    /** Resize the list samples so that enough memory is allocated. */
    listSampleFixed->SetMeasurementVectorSize( fixedSize );
    listSampleFixed->Resize( size );
    listSampleMoving->SetMeasurementVectorSize( movingSize );
    listSampleMoving->Resize( size );
    listSampleJoint->SetMeasurementVectorSize( jointSize );
    listSampleJoint->Resize( size );

    /** Create variables to store intermediate results. */
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    double fixedFeatureValue = 0.0;
    double movingFeatureValue = 0.0;

    /** Loop over the fixed image samples to calculate the list samples. */
    unsigned int ii = 0;
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
      /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;

      /** Transform point and check if it is inside the B-spline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside all moving masks. */
      if ( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );        
      }

      /** Compute the moving image value M(T(x)) and possibly the
       * derivative dM/dx and check if the point is inside all
       * moving images buffers.
       */
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
          this->GetNumberOfFixedImages(), movingImageValue );

        /** Get and set the values of the fixed feature images. */
        for ( unsigned int j = 1; j < this->GetNumberOfFixedImages(); j++ )
        {
          fixedFeatureValue = this->m_FixedImageInterpolatorVector[ j ]->Evaluate( fixedPoint );
          listSampleFixed->SetMeasurement(
            this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
          listSampleJoint->SetMeasurement(
            this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
        }

        /** Get and set the values of the moving feature images. */
        for ( unsigned int j = 1; j < this->GetNumberOfMovingImages(); j++ )
        {
          movingFeatureValue = this->m_InterpolatorVector[ j ]->Evaluate( mappedPoint );
          listSampleMoving->SetMeasurement(
            this->m_NumberOfPixelsCounted,
            j,
            movingFeatureValue );
          listSampleJoint->SetMeasurement(
            this->m_NumberOfPixelsCounted,
            j + this->GetNumberOfFixedImages(),
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
            this->GetNumberOfMovingImages(),
            this->FixedImageDimension );
          spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

          /** Get the spatial derivatives of the moving feature images. */
          SpatialDerivativeType movingFeatureImageDerivatives(
            this->GetNumberOfMovingImages() - 1,
            this->FixedImageDimension );
          this->EvaluateMovingFeatureImageDerivatives(
            mappedPoint, movingFeatureImageDerivatives );
          spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

          /** Put the spatial derivatives of this sample into the container. */
          spatialDerivativesContainer.push_back( spatialDerivatives );

        } // end if doDerivative
       
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

        ii++;

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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
  ::EvaluateMovingFeatureImageDerivatives(
    const MovingImagePointType & mappedPoint,
    SpatialDerivativeType & featureGradients ) const
  {
    /** Convert point to a continous index. */
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );

    /** Compute the spatial derivative for all feature images:
     * - either by calling a special function that only B-spline
     *   interpolators have,
     * - or by using a finite difference approximation of the
     *   pre-computed gradient images.
     * \todo: for now we only implement the first option.
     */
    if ( this->m_InterpolatorsAreBSpline && !this->GetComputeGradient() )
    {
      /** Computed moving image gradient using derivative BSpline kernel. */
      MovingImageDerivativeType gradient;
      for ( unsigned int i = 1; i < this->GetNumberOfMovingImages(); ++i )
      {
        /** Compute the gradient at feature image i. */
        gradient = this
          ->m_BSplineInterpolatorVector[ i ]
          ->EvaluateDerivativeAtContinuousIndex( cindex );

        /** Set the gradient into the Array2D. */
        featureGradients.set_row( i - 1, gradient.GetDataPointer() );
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
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
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
	 * ************************ ComputeImageJacobianDifference2 *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
  ::ComputeImageJacobianDifference2(
    const SpatialDerivativeType & D1sparse,
    const SpatialDerivativeType & D2sparse_M,
    const SpatialDerivativeType & D2sparse_J,
    const ParameterIndexArrayType & D1indices,
    const ParameterIndexArrayType & D2indices_M,
    const ParameterIndexArrayType & D2indices_J,
    const MeasurementVectorType & diff_M,
    const MeasurementVectorType & diff_J,
    const MeasureType distance_M,
    const MeasureType distance_J,
    DerivativeType & dGamma_M,
    DerivativeType & dGamma_J ) const
  {
		/** Make temporary copies of diff, since post_multiply changes diff. */
		vnl_vector<double> tmpM1( diff_M );
		vnl_vector<double> tmpM2( diff_M );
		vnl_vector<double> tmpJ( diff_J );

		/** Compute sparse intermediary results. */
    vnl_vector<double> tmp1sparse   = tmpM1.post_multiply( D1sparse );
    vnl_vector<double> tmp2sparse_M = tmpM2.post_multiply( D2sparse_M );
    vnl_vector<double> tmp2sparse_J = tmpJ.post_multiply( D2sparse_J );

    /** Add first half. *
    for ( unsigned int i = 0; i < D1indices.GetSize(); ++i )
    {
      dGamma_M[ D1indices[ i ] ] += tmp1sparse[ i ];
      dGamma_J[ D1indices[ i ] ] += tmp1sparse[ i ];
    }*/

    if ( distance_M > this->m_AvoidDivisionBy )
    {
      for ( unsigned int i = 0; i < D1indices.GetSize(); ++i )
      {
        dGamma_M[ D1indices[ i ] ] += tmp1sparse[ i ];
      }
    }
    if ( distance_J > this->m_AvoidDivisionBy )
    {
      for ( unsigned int i = 0; i < D1indices.GetSize(); ++i )
      {
        dGamma_J[ D1indices[ i ] ] += tmp1sparse[ i ];
      }
    }
    
    /** Subtract second half. */
    if ( distance_M > this->m_AvoidDivisionBy )
    {
      for ( unsigned int i = 0; i < D2indices_M.GetSize(); ++i )
      {
        dGamma_M[ D2indices_M[ i ] ] -= tmp2sparse_M[ i ];
      }
    }

    if ( distance_J > this->m_AvoidDivisionBy )
    {
      for ( unsigned int i = 0; i < D2indices_J.GetSize(); ++i )
      {
        dGamma_J[ D2indices_J[ i ] ] -= tmp2sparse_J[ i ];
      }
    }

  } // end ComputeImageJacobianDifference2()


  /**
	 * ************************ PrintSelf *************************
	 */
  
  template <class TFixedImage, class TMovingImage>
  void
  KNNGraphAlphaMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
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


#endif // end #ifndef _itkKNNGraphAlphaMutualInformationImageToImageMetric_txx

