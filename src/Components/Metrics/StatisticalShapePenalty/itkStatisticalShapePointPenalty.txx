/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

  If you use the StatisticalShapePenalty anywhere we would appreciate if you cite the following article:
  F.F. Berendsen et al., Free-form image registration regularized by a statistical shape model: application to organ segmentation in cervical MR, Comput. Vis. Image Understand. (2013), http://dx.doi.org/10.1016/j.cviu.2012.12.006

======================================================================*/
#ifndef __itkStatisticalShapePointPenalty_txx
#define __itkStatisticalShapePointPenalty_txx

#include "itkStatisticalShapePointPenalty.h"


namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::StatisticalShapePointPenalty()
{
    m_MeanVector = NULL;
    m_EigenVectors = NULL;
    m_EigenValues = NULL;
    m_EigenValuesRegularized = NULL;
    m_ProposalDerivative = NULL;
    m_InverseCovarianceMatrix = NULL;

    m_ShrinkageIntensityNeedsUpdate = true;
    m_BaseVarianceNeedsUpdate = true;
    m_VariancesNeedsUpdate = true;   
  //this->SetUseImageSampler( false );
} // end Constructor

/**
 * ******************* Destructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::~StatisticalShapePointPenalty()
{
  if (m_MeanVector!=NULL){
    delete m_MeanVector;
    m_MeanVector=NULL;
  }
   if (m_CovarianceMatrix!=NULL){
    delete m_CovarianceMatrix;
    m_CovarianceMatrix=NULL;
  }
  if (m_EigenVectors!=NULL){
    delete m_EigenVectors;
    m_EigenVectors=NULL;
  }
  if (m_EigenValues!=NULL){
    delete m_EigenValues;
    m_EigenValues=NULL;
  }
  if (m_EigenValuesRegularized!=NULL){
    delete m_EigenValuesRegularized;
    m_EigenValuesRegularized=NULL;
  } 
  if ( m_ProposalDerivative!=NULL){
    delete  m_ProposalDerivative;
   m_ProposalDerivative=NULL;
  } 
  if ( m_InverseCovarianceMatrix!=NULL){
    delete  m_InverseCovarianceMatrix;
   m_InverseCovarianceMatrix=NULL;
  } 

  

} // end Destructor

/**
* *********************** Initialize *****************************
*/

template< class TFixedPointSet, class TMovingPointSet >
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the initialize of the superclass. */
  this->Superclass::Initialize();

  const unsigned int shapeLength = Self::FixedPointSetDimension*(this->GetFixedPointSet()->GetNumberOfPoints());
  if (m_NormalizedShapeModel)
  {
    m_ProposalLength = shapeLength+Self::FixedPointSetDimension+1;// [[normalized shape],[centroid],l2norm]

    /** Automatic selection of regularization variances. */
    if (m_BaseVariance==-1.0 || m_CentroidXVariance==-1.0 || m_CentroidYVariance==-1.0 || m_CentroidZVariance==-1.0 || m_SizeVariance ==-1.0){
      vnl_vector<double> covDiagonal = m_CovarianceMatrix->get_diagonal();
      if (m_BaseVariance==-1.0){
        m_BaseVariance = covDiagonal.extract(shapeLength).mean();
        elxout << "Automatic selection of BaseVariance: " << m_BaseVariance << std::endl;
      }
      if (m_CentroidXVariance==-1.0){
        m_CentroidXVariance = covDiagonal.get(shapeLength);
        elxout << "Automatic selection of CentroidXVariance: " << m_CentroidXVariance << std::endl;
      }
      if (m_CentroidYVariance==-1.0){
        m_CentroidYVariance = covDiagonal.get(shapeLength+1);
        elxout << "Automatic selection of CentroidYVariance: " << m_CentroidYVariance << std::endl;
      }
      if (m_CentroidZVariance==-1.0){
        m_CentroidZVariance = covDiagonal.get(shapeLength+2);
        elxout << "Automatic selection of CentroidZVariance: " << m_CentroidZVariance << std::endl;
      }
      if (m_SizeVariance==-1.0){
        m_SizeVariance = covDiagonal.get(shapeLength+3);
        elxout << "Automatic selection of SizeVariance: " << m_SizeVariance << std::endl;
      }

    } // End automatic selection of regularization variances.
  }
  else
  {
    m_ProposalLength = shapeLength;
    /** Automatic selection of regularization variances. */
    if (m_BaseVariance==-1.0){
      vnl_vector<double> covDiagonal = m_CovarianceMatrix->get_diagonal();
      m_BaseVariance = covDiagonal.extract(shapeLength).mean();
      elxout << "Automatic selection of BaseVariance: " << m_BaseVariance << std::endl;
    }// End automatic selection of regularization variances.

  }
  //this->m_proposalvector.set_size(m_ProposalLength); 
  //m_ProposalDerivative = ProposalDerivativeType(this->GetNumberOfParameters());
  //m_ProposalDerivative = ProposalDerivativeType::New(this->GetNumberOfParameters());



  switch ( m_ShapeModelCalculation )
  {
  case 0: // full covariance 
    {
      
      if(m_ShrinkageIntensityNeedsUpdate || m_BaseVarianceNeedsUpdate || (m_NormalizedShapeModel && m_VariancesNeedsUpdate ))
      {
        vnl_matrix< double > regularizedCovariance = (1-m_ShrinkageIntensity) * (*m_CovarianceMatrix);
        vnl_vector<double> regCovDiagonal = regularizedCovariance.get_diagonal();
        if (m_NormalizedShapeModel)
        {
          regCovDiagonal.update(m_ShrinkageIntensity*m_BaseVariance+regCovDiagonal.extract(shapeLength));
          regCovDiagonal[shapeLength]+=m_ShrinkageIntensity*m_CentroidXVariance;
          regCovDiagonal[shapeLength+1]+=m_ShrinkageIntensity*m_CentroidYVariance;
          regCovDiagonal[shapeLength+2]+=m_ShrinkageIntensity*m_CentroidZVariance;
          regCovDiagonal[shapeLength+3]+=m_ShrinkageIntensity*m_SizeVariance;

        }
        else
        {
          regCovDiagonal += m_ShrinkageIntensity*m_BaseVariance;
        }
        regularizedCovariance.set_diagonal(regCovDiagonal);
        /** If no regularization is applied, the user is responsible for providing an invertible Covariance Matrix.
            For a Moore-Penrose pseudo inverse use ShrinkageIntensity=0 and ShapeModelCalculation=1 or 2.
        */
        m_InverseCovarianceMatrix = new vnl_matrix<double>(vnl_svd_inverse(regularizedCovariance));
      }
      m_EigenValuesRegularized=NULL;
      break;
    }
  case 1: // decomposed covariance (uniform regularization)
    {
      if(m_NormalizedShapeModel==true){
        itkExceptionMacro( << "ShapeModelCalculation option 1 is only implemented for NormalizedShapeModel = false" );
      }

      PCACovarianceType pcaCovariance(*m_CovarianceMatrix);
      typename VnlVectorType::iterator lambdaIt = pcaCovariance.lambdas().begin();
      typename VnlVectorType::iterator lambdaEnd = pcaCovariance.lambdas().end();
      unsigned int nonZeroLength=0;
      for (;lambdaIt != lambdaEnd && (*lambdaIt)>1e-14; ++lambdaIt, ++nonZeroLength){}
      elxout << "Number of non-zero eigenvalues: " << nonZeroLength << std::endl;
      if (m_EigenValues!=NULL){
        delete m_EigenValues;
      }
      m_EigenValues = new VnlVectorType(pcaCovariance.lambdas().extract(nonZeroLength));

      if (m_EigenVectors!=NULL){
        delete m_EigenVectors;
      }
      m_EigenVectors = new VnlMatrixType(pcaCovariance.V().get_n_columns(0,nonZeroLength));


      if (m_EigenValuesRegularized==NULL)
      {
        m_EigenValuesRegularized = new vnl_vector<double>(m_EigenValues->size());
      }
      //m_PCACovariance->V();
      elxout << m_EigenVectors->rows() << std::endl;

      vnl_vector<double>::iterator regularizedValue;
      vnl_vector<double>::const_iterator eigenValue;

      //vnl_svd_economy<double> pca(*m_CovarianceMatrix);
      if (m_ShrinkageIntensity!=0)
        // if there is regularization (>0), the eigenvalues are altered and stored in regularizedValue
      {
        for( regularizedValue = m_EigenValuesRegularized->begin(),
          eigenValue = m_EigenValues->begin();
          regularizedValue!= m_EigenValuesRegularized->end(); 
        regularizedValue++, eigenValue++)
        {
          *regularizedValue= - m_ShrinkageIntensity * m_BaseVariance - m_ShrinkageIntensity * m_BaseVariance * m_ShrinkageIntensity * m_BaseVariance / (1.0-m_ShrinkageIntensity) / *eigenValue;

        }
      }else 
        /* If there is no regularization (m_ShrinkageIntensity==0), a division by zero is avoided by just copying the eigenvalues to regularizedValue. 
        However this will be handled correctly in the calculation of the value and derivative. 
        Providing a non-square eigenvector matrix, with associated eigen values that are non-zero yields a Mahalanobis distance calculation with a pseudo         inverse.
        */

      {
        for( regularizedValue = m_EigenValuesRegularized->begin(),
          eigenValue = m_EigenValues->begin();
          regularizedValue!= m_EigenValuesRegularized->end(); 
        regularizedValue++, eigenValue++)
        {
          *regularizedValue= *eigenValue;
        }
      }

      m_InverseCovarianceMatrix = NULL;       
    }
    break;
  case 2: // decomposed scaled covariance (element specific regularization)
    {
      if( m_NormalizedShapeModel== false )
      {
        itkExceptionMacro( << "ShapeModelCalculation option 2 is only implemented for NormalizedShapeModel = true" );
      }

      bool pcaNeedsUpdate = false;

      if( m_BaseVarianceNeedsUpdate || m_VariancesNeedsUpdate){
        pcaNeedsUpdate= true;
        m_BaseStd = sqrt(m_BaseVariance);
        m_CentroidXStd = sqrt(m_CentroidXVariance);
        m_CentroidYStd = sqrt(m_CentroidYVariance);
        m_CentroidZStd = sqrt(m_CentroidZVariance);
        m_SizeStd = sqrt(m_SizeVariance);
        vnl_matrix< double > scaledCovariance(*m_CovarianceMatrix);

        scaledCovariance.set_columns(0,scaledCovariance.get_n_columns(0,shapeLength)/m_BaseStd);
        scaledCovariance.scale_column(shapeLength, 1.0/m_CentroidXStd);
        scaledCovariance.scale_column(shapeLength+1, 1.0/m_CentroidYStd);
        scaledCovariance.scale_column(shapeLength+2, 1.0/m_CentroidZStd);
        scaledCovariance.scale_column(shapeLength+3, 1.0/m_SizeStd);


        //scaledCovariance.set_rows(0,scaledCovariance.get_n_rows(0,shapeLength)/m_BaseStd);
        scaledCovariance.update(scaledCovariance.get_n_rows(0,shapeLength)/m_BaseStd);

        scaledCovariance.scale_row(shapeLength, 1.0/m_CentroidXStd);
        scaledCovariance.scale_row(shapeLength+1, 1.0/m_CentroidYStd);
        scaledCovariance.scale_row(shapeLength+2, 1.0/m_CentroidZStd);
        scaledCovariance.scale_row(shapeLength+3, 1.0/m_SizeStd);

        PCACovarianceType pcaCovariance(scaledCovariance);
        typename VnlVectorType::iterator lambdaIt = pcaCovariance.lambdas().begin();
        typename VnlVectorType::iterator lambdaEnd = pcaCovariance.lambdas().end();
        unsigned int nonZeroLength=0;
        for (;lambdaIt != lambdaEnd && (*lambdaIt)>1e-14; ++lambdaIt, ++nonZeroLength){}
        elxout << "Number of non-zero eigenvalues: " << nonZeroLength << std::endl;
        if (m_EigenValues!=NULL){
          delete m_EigenValues;
        }
        m_EigenValues = new VnlVectorType(pcaCovariance.lambdas().extract(nonZeroLength));

        if (m_EigenVectors!=NULL){
          delete m_EigenVectors;
        }
        m_EigenVectors = new VnlMatrixType(pcaCovariance.V().get_n_columns(0,nonZeroLength));
      }
      if (m_ShrinkageIntensityNeedsUpdate || pcaNeedsUpdate)
      {
        if (m_EigenValuesRegularized!=NULL)
        {
          delete m_EigenValuesRegularized;
        }
        if (m_ShrinkageIntensity!=0)
          // if there is regularization (>0), the eigenvalues are altered and kept in regularizedValue
        {
          m_EigenValuesRegularized = new vnl_vector<double>(m_EigenValues->size());
          typename vnl_vector<double>::iterator regularizedValue;
          typename vnl_vector<double>::const_iterator eigenValue;
          for( regularizedValue = m_EigenValuesRegularized->begin(),
            eigenValue = m_EigenValues->begin();
            regularizedValue!= m_EigenValuesRegularized->end(); 
          regularizedValue++, eigenValue++)
          {
            *regularizedValue= - m_ShrinkageIntensity  - m_ShrinkageIntensity * m_ShrinkageIntensity / (1.0-m_ShrinkageIntensity) / *eigenValue;

          }
        }else 
          /* If there is no regularization (m_ShrinkageIntensity==0), a division by zero is avoided by just copying the eigenvalues to regularizedValue. 
          However this will be handled correctly in the calculation of the value and derivative. 
          Providing a non-square eigenvector matrix, with associated eigen values that are non-zero yields a Mahalanobis distance calculation with a pseudo         inverse.
          */

        {
          m_EigenValuesRegularized = new VnlVectorType(*m_EigenValues);
        }

      }
      m_ShrinkageIntensityNeedsUpdate = false;
      m_BaseVarianceNeedsUpdate = false;
      m_VariancesNeedsUpdate = false;

      m_InverseCovarianceMatrix = NULL;
    }
    break;
  default:
    m_InverseCovarianceMatrix = NULL;
    m_EigenValuesRegularized= NULL;

    elxout << "Error: Bad m_ShapeModelCalculation" << std::endl;
  }


} // end Initialize()


/**
 * ******************* GetValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
typename StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>::MeasureType
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::GetValue( const TransformParametersType & parameters ) const
{
    /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
  {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
  }

  /** Initialize some variables */
  //this->m_NumberOfPointsCounted = 0;
  MeasureType value = NumericTraits< MeasureType >::Zero;

  //InputPointType movingPoint;
  OutputPointType fixedPoint;
  /** Get the current corresponding points. */
  
  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  const unsigned int shapeLength = Self::FixedPointSetDimension*(fixedPointSet->GetNumberOfPoints());
  //const unsigned int proposalLength = shapeLength+FixedPointSetDimension+1;
  this->m_proposalvector.set_size(m_ProposalLength); // [[normalized shape],[centroid],l2norm]

/** Part 1: 
  - Copy point positions in proposal vector
  */

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex=0;
  /** Loop over the corresponding points. */
  while( pointItFixed != pointEnd )
  {
    fixedPoint = pointItFixed.Value();
    this->FillProposalVector(fixedPoint,vertexindex);

    //std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
    this->m_NumberOfPointsCounted++;
    ++pointItFixed;
    vertexindex+=Self::FixedPointSetDimension;
  } // end loop over all corresponding points

  if (m_NormalizedShapeModel)
  {  
  /** Part 2: 
    - Calculate shape centroid 
      - put centroid values in proposal
      - update proposal vector with aligned shape
  */
    this->UpdateCentroidAndAlignProposalVector(shapeLength);

    /** Part 3:
      - Calculate l2-norm from aligned shapes 
        - put l2-norm value in proposal vector
        - update proposal vector with size normalized shape
       */
    
    this->UpdateL2(shapeLength);
    this->NormalizeProposalVector(shapeLength);
  }
  VnlVectorType differenceVector;
  VnlVectorType centerrotated;
  VnlVectorType eigrot;

  this->CalculateValue(value,differenceVector,centerrotated,eigrot);

  return value;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::GetDerivative( const TransformParametersType & parameters,
                DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
  * the metric value is available. It does not cost anything to calculate
  * the metric value now. Therefore, we have chosen to only implement the
  * GetValueAndDerivative(), supplying it with a dummy value variable.
  */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
* ******************* GetValueAndDerivative *******************
*/

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType & value, DerivativeType & derivative ) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
  {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
  }

  /** Initialize some variables */
  //this->m_NumberOfPointsCounted = 0;
  //MeasureType 
  value = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  //InputPointType movingPoint;
  OutputPointType fixedPoint;
  /** Get the current corresponding points. */

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  const unsigned int shapeLength = Self::FixedPointSetDimension*(fixedPointSet->GetNumberOfPoints());
  //const unsigned int proposalLength = shapeLength+FixedPointSetDimension+1;
  //this->m_proposalvector.set_size(proposalLength); // [[normalized shape],[centroid],l2norm]

  //m_ProposalDerivative = ProposalDerivativeType(this->GetNumberOfParameters(),NULL);
  this->m_proposalvector.set_size(m_ProposalLength); 
  //std::fill(m_ProposalDerivative->begin(), m_ProposalDerivative->end(), NULL);
  m_ProposalDerivative = new ProposalDerivativeType(this->GetNumberOfParameters(),NULL);
  //m_ProposalDerivative->Fill(NULL);
  /** Part 1: 
  - Copy point positions in proposal vector
  - Copy point derivatives in proposal derivative vector */

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex=0;
  /** Loop over the corresponding points. */
  while( pointItFixed != pointEnd )
  {
    fixedPoint = pointItFixed.Value();
    this->FillProposalVector(fixedPoint,vertexindex);
    this->FillProposalDerivative(fixedPoint,vertexindex);

    //std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
    this->m_NumberOfPointsCounted++;
    ++pointItFixed;
    vertexindex+=Self::FixedPointSetDimension;
  } // end loop over all corresponding points

  if (m_NormalizedShapeModel)
  {
    /** Part 2: 
    - Calculate shape centroid 
    - put centroid values in proposal
    - update proposal vector with aligned shape
    - Calculate centroid derivatives and update proposal derivative vectors 
    - put centroid derivatives values in proposal derivatives
    - update proposal derivatives */
    this->UpdateCentroidAndAlignProposalVector(shapeLength);
    this->UpdateCentroidAndAlignProposalDerivative(shapeLength);

    /** Part 3:
    - Calculate l2-norm from aligned shapes 
    - put l2-norm value in proposal vector
    - update proposal vector with size normalized shape
    - Calculate l2-norm derivatice from updated proposal vector
    - put l2-norm derivative value in proposal derivative vectors
    - update proposal derivatives */

    this->UpdateL2(shapeLength);
    this->UpdateL2AndNormalizeProposalDerivative(shapeLength);
    this->NormalizeProposalVector(shapeLength);

  } // end if(m_NormalizedShapeModel)
  VnlVectorType differenceVector; // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.
  VnlVectorType centerrotated; // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.
  VnlVectorType eigrot; // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.

  this->CalculateValue(value,differenceVector,centerrotated,eigrot);

  if (value!=0.0)
  {
    this->CalculateDerivative(derivative,value,differenceVector,centerrotated,eigrot,shapeLength);
  }
  else
  {
    typename ProposalDerivativeType::iterator proposalDerivativeIt = m_ProposalDerivative->begin();
    typename ProposalDerivativeType::iterator proposalDerivativeEnd = m_ProposalDerivative->end();
    for(;proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt)
      {
      if(*proposalDerivativeIt!=NULL)
      {
        delete (*proposalDerivativeIt);
      }
    }
  }
  delete m_ProposalDerivative;
  m_ProposalDerivative=NULL;
 
  CalculateCutOffValue(value);
  //elxout << "non zero jacobian Shape columns: " << nzjacs << std::endl;

  //elxout << "derivatives: " << derivative << std::endl;

} // end GetValueAndDerivative()


template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::FillProposalVector(const OutputPointType & fixedPoint, const unsigned int vertexindex) const
{
  OutputPointType mappedPoint;
  /** Get the current corresponding points. */
  //meanPoint = pointItMean.Value();
  mappedPoint = this->m_Transform->TransformPoint( fixedPoint );
  //elxout << "TransformPoint oke" << std::endl;

  /** Transform point and check if it is inside the B-spline support region. */
  //bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );


  /** copy n-D coordinates into big Shape vector. Aligning the centroids is done later*/
  for( unsigned int d=0; d< Self::FixedPointSetDimension; ++d){
    this->m_proposalvector[vertexindex+d]=mappedPoint[d]; 
    //elxout << "copy oke" << std::endl;
  }
}// end FillProposalVector()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::FillProposalDerivative(const OutputPointType & fixedPoint, const unsigned int vertexindex) const
{
  /*
  A (column) vector is constructed for each mu, only if that mu affects the shape penalty. I.e. if there is at least one point of the mesh with non-zero derivatives, a full column vector is instantiated (which can contain zeros for many other points)

  m_ProposalDerivative is a container with either full shape-vector-sized derivative vectors or NULL-s. Example:

  mu1: [ [ dx1/dmu1 , dy1/dmu1 , dz1/dmu1 ] , [ 0 , 0 , 0 ] , [ dx3/dmu1 , dy3/dmu1 , dz3/dmu1 ] , [...] ]^T
  mu2: Null 
  mu3: [ [ 0 , 0 , 0 ] , [ dx2/dmu3 , dy2/dmu3 , dz2/dmu3 ] , [ dx3/dmu3 , dy3/dmu3 , dz3/dmu3 ] , [...] ]^T
  ...

  */

  NonZeroJacobianIndicesType nzji(
    this->m_Transform->GetNumberOfNonZeroJacobianIndices() );

  /** Get the TransformJacobian dT/dmu. */
  TransformJacobianType jacobian;
  //this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
  this->m_Transform->GetJacobian( fixedPoint, jacobian, nzji );
  //elxout << "GetJacobian oke" << std::endl;

  for ( unsigned int i = 0; i < nzji.size(); ++i )
  {
    const unsigned int mu = nzji[ i ];
    //elxout << "non zero jac index:" << index << std::endl;
    if ((*m_ProposalDerivative)[mu]==NULL)
    {
      /** Create the big column vector if it does not yet exist for this mu*/
      (*m_ProposalDerivative)[mu]=new VnlVectorType(this->m_ProposalLength,0.0);
      // memory will be freed in CalculateDerivative()
      //elxout << "new muColumn oke" << std::endl;
    }

    /** The column vector exists for this mu, so copy the jacobians for this point into the big vector*/
    //elxout << "jacobian column: " << jacobian.get_column( i ) << std::endl;
    //elxout << "mu column: " << (*(muColumn[index]))[vertexindex*3] << std::endl;
    for( unsigned int d=0; d< Self::FixedPointSetDimension; ++d){
      (*((*m_ProposalDerivative)[mu]))[ vertexindex + d ]=jacobian.get_column( i )[d];
    }      //elxout << "copy jacobian oke" << std::endl;

    //nzjiflags[index] = TRUE;
  }
}// end FillProposalVector()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::UpdateCentroidAndAlignProposalVector(const unsigned int shapeLength) const{
  //double &centroidx = m_proposalvector[shapeLength];
  //double &centroidy = m_proposalvector[shapeLength+1];
  //double &centroidz = m_proposalvector[shapeLength+2];

  //double* const centroid = &(m_proposalvector[shapeLength]); // Create an alias for the centroid elements in the proposal vector

  /** Aligning Shapes with their centroids */
  for( unsigned int d=0; d< Self::FixedPointSetDimension; ++d){ // loop over dimension x,y,z

    double &centroid_d = m_proposalvector[shapeLength+d]; // Create an alias for the centroid elements in the proposal vector

    centroid_d=0; // initialize centroid x,y,z to zero

    for(unsigned int index=0; index < shapeLength ; index+=Self::FixedPointSetDimension ){
      centroid_d+=m_proposalvector[index+d];; // sum all x coordinates to centroid_x, y to centroid_y ...
    }

    centroid_d/=this->GetFixedPointSet()->GetNumberOfPoints();  // divide sum to get average

    for(unsigned int index=0; index < shapeLength ; index+=Self::FixedPointSetDimension ){
      // subtract average
      m_proposalvector[index+d]-=centroid_d; 
    }
  }
} // end UpdateCentroidAndAlignProposalVector()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::UpdateCentroidAndAlignProposalDerivative(const unsigned int shapeLength) const{

  typename ProposalDerivativeType::iterator proposalDerivativeIt = m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = m_ProposalDerivative->end();
  while(proposalDerivativeIt != proposalDerivativeEnd)
  {
    if(*proposalDerivativeIt!=NULL)
    {
      for( unsigned int d=0; d< Self::FixedPointSetDimension; ++d){ // loop over dimension x,y,z
        double& centroid_dDerivative = (**proposalDerivativeIt)[shapeLength+d];
        //double& centroidyDerivative = (**proposalDerivativeIt)[shapeLength+1];
        //double& centroidzDerivative = (**proposalDerivativeIt)[shapeLength+2];
        centroid_dDerivative=0; // initialize accumulators to zero
        //centroidyDerivative=0; 
        //centroidzDerivative=0; 
        for(unsigned int index=0; index < shapeLength ; index+=Self::FixedPointSetDimension ){
          centroid_dDerivative+=(**proposalDerivativeIt)[index+d]; // sum all x derivatives
          //centroidyDerivative+=(**proposalDerivativeIt)[index+1];
          //centroidzDerivative+=(**proposalDerivativeIt)[index+2];
        }

        centroid_dDerivative/=this->GetFixedPointSet()->GetNumberOfPoints(); // divide sum to get average
        //centroidyDerivative/=shapeLength;
        //centroidzDerivative/=shapeLength;

        for(unsigned int index=0; index < shapeLength ; index+=Self::FixedPointSetDimension ){
          (**proposalDerivativeIt)[index+d]-=centroid_dDerivative; // subtract average
          //(**proposalDerivativeIt)[index+1]-=centroidyDerivative;
          //(**proposalDerivativeIt)[index+2]-=centroidzDerivative;
        }
      }
    }
    ++proposalDerivativeIt;
  }   
} // end UpdateCentroidAndAlignProposalDerivative()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::UpdateL2(const unsigned int shapeLength) const{
  double& l2norm = m_proposalvector[shapeLength + Self::FixedPointSetDimension]; 

  l2norm=0; // initialize l2norm to zero
  for(unsigned int index=0; index<shapeLength ; index++){ // loop over all shape coordinates of the aligned shape
    l2norm+=m_proposalvector[index]*m_proposalvector[index]; // accumulate squared distances
  }
  l2norm=sqrt(l2norm/this->GetFixedPointSet()->GetNumberOfPoints()); // 

} // end UpdateL2AndNormalizeProposalVector()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::NormalizeProposalVector(const unsigned int shapeLength) const{
  double& l2norm = m_proposalvector[shapeLength + Self::FixedPointSetDimension]; 

  for(unsigned int index=0; index<shapeLength ; index++){ // loop over all shape coordinates of the aligned shape
    m_proposalvector[index]/=l2norm ; // normalize shape size by l2-norm
  }

} // end UpdateL2AndNormalizeProposalVector()
template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::UpdateL2AndNormalizeProposalDerivative(const unsigned int shapeLength) const{

  double& l2norm = m_proposalvector[shapeLength + Self::FixedPointSetDimension]; 

  typename ProposalDerivativeType::iterator proposalDerivativeIt = m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = m_ProposalDerivative->end();

  while(proposalDerivativeIt != proposalDerivativeEnd)
  {
    if(*proposalDerivativeIt!=NULL)
    {
      double& l2normDerivative = (**proposalDerivativeIt)[shapeLength + Self::FixedPointSetDimension];
      l2normDerivative=0; // initialize to zero
      for(unsigned int index=0; index<shapeLength ; index++){ // loop over all shape coordinates of the aligned shape
        l2normDerivative+=m_proposalvector[index]*(**proposalDerivativeIt)[index];
      }
      l2normDerivative/=(l2norm*sqrt((double)(this->GetFixedPointSet()->GetNumberOfPoints())));
      for(unsigned int index=0; index<shapeLength ; index++){ // loop over all shape coordinates of the aligned shape
        (**proposalDerivativeIt)[index]= (**proposalDerivativeIt)[index] / l2norm - m_proposalvector[index] * l2normDerivative / (l2norm*l2norm); // update normalized shape derivatives
      }

    }
    ++proposalDerivativeIt;
  }   
} //end UpdateL2AndNormalizeProposalDerivative()


template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::CalculateValue(MeasureType & value , VnlVectorType & differenceVector , VnlVectorType & centerrotated , VnlVectorType & eigrot) const{

  differenceVector = (m_proposalvector- *m_MeanVector);

  switch ( m_ShapeModelCalculation )
  {
  case 0: // full covariance 
    {
      //value = dot_product(differenceVector, *m_InverseCovarianceMatrix); 
      //value = sqrt(differenceVector * *m_InverseCovarianceMatrix * differenceVector );
       value = sqrt(bracket(differenceVector,*m_InverseCovarianceMatrix,differenceVector));
      break;
    }
  case 1: // decomposed covariance (uniform regularization)
    {
      centerrotated = differenceVector * (*m_EigenVectors); /** diff^T * V */
      // elxout << "centered: " << m_proposalvector << std::endl;
      //elxout << "rotated: " << centerrotated << std::endl;
      //elxout << "eigenvalues: " <<*m_EigenValuesRegularized << std::endl;
      eigrot = element_quotient(centerrotated,*m_EigenValuesRegularized); /** diff^T * V * Lambda^-1 */
      //elxout << "rotated./eigenvalues: " << eigrot << std::endl;
      if (m_ShrinkageIntensity!=0)
      {
        //differenceVector.magnitude();
        value = sqrt(dot_product(eigrot, centerrotated) + dot_product(differenceVector,differenceVector)/(m_ShrinkageIntensity * m_BaseVariance)); /** innerproduct diff^T * V * Lambda^-1 * V^T * diff  +  1/(sigma_0*Beta)* diff^T*diff*/
      }
      else
      {
        value = sqrt(dot_product(eigrot, centerrotated)); /** innerproduct diff^T * V * Lambda^-1 * V^T * diff*/
      }

      break;
    }
  case 2: // decomposed scaled covariance (element specific regularization)
    {
      const unsigned int shapeLength=m_ProposalLength - Self::FixedPointSetDimension -1;
      typename VnlVectorType::iterator diffElementIt = differenceVector.begin();
      for( int diffElementIndex=0; diffElementIndex<shapeLength; ++diffElementIndex, ++diffElementIt){
        (*diffElementIt)/=m_BaseStd;
      }
      differenceVector[shapeLength]/=m_CentroidXStd;
      differenceVector[shapeLength+1]/=m_CentroidYStd;
      differenceVector[shapeLength+2]/=m_CentroidZStd;
      differenceVector[shapeLength+3]/=m_SizeStd;
 
      centerrotated = differenceVector * (*m_EigenVectors); /** diff^T * V */
      eigrot = element_quotient(centerrotated,*m_EigenValuesRegularized); /** diff^T * V * Lambda^-1 */
      if (m_ShrinkageIntensity!=0)
      {
        //differenceVector.magnitude();
        //differenceVector.squared_magnitude()
        value = sqrt(dot_product(eigrot, centerrotated) + differenceVector.squared_magnitude()/m_ShrinkageIntensity); /** innerproduct diff^T * ~V * I * ~V^T * diff  +  1/(Beta)* diff^T*diff*/
      }
      else
      {
        value = sqrt(dot_product(eigrot, centerrotated)); /** innerproduct diff^T * V * I * V^T * diff*/
      }

      break;
    }
  default:
    elxout << "Error: Bad m_ShapeModelCalculation" << std::endl;
  }


} //end CalculateValue() 

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::CalculateDerivative(DerivativeType & derivative, const MeasureType & value, const VnlVectorType & differenceVector , const VnlVectorType & centerrotated , const VnlVectorType & eigrot, const unsigned int shapeLength) const{

  typename ProposalDerivativeType::iterator proposalDerivativeIt = m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = m_ProposalDerivative->end();

  typename DerivativeType::iterator derivativeIt = derivative.begin();
  //unsigned int nzjacs = 0;

  for(;proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt, ++derivativeIt)
  {
    if(*proposalDerivativeIt!=NULL)
    {
      switch ( m_ShapeModelCalculation )
      {
      case 0: // full covariance 
        {
          /**innerproduct diff^T * Sigma^-1 * d/dmu (diff), where iterated over mu-s*/
          *derivativeIt = bracket(differenceVector,*m_InverseCovarianceMatrix,(**proposalDerivativeIt))/value;
          CalculateCutOffDerivative(*derivativeIt,value);
          break;
        }
      case 1: // decomposed covariance (uniform regularization)
        {
          if (m_ShrinkageIntensity!=0)
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu(diff)+ 1/(Beta*sigma_0^2)*diff^T* d/dmu(diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot,m_EigenVectors->transpose()*(**proposalDerivativeIt))+ dot_product(differenceVector,**proposalDerivativeIt)/(m_ShrinkageIntensity * m_BaseVariance))/value; 
            CalculateCutOffDerivative(*derivativeIt,value);
          }
          else //m_ShrinkageIntensity==0
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu (diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot,m_EigenVectors->transpose()*(**proposalDerivativeIt)))/value; 
            CalculateCutOffDerivative(*derivativeIt,value);
          }
          break;
        }
      case 2: // decomposed scaled covariance (element specific regularization)
        {
          // first scale proposalDerivatives with their sigma's in order to evaluate with the EigenValues and EigenVectors of the scaled               CovarianceMatrix
          typename VnlVectorType::iterator propDerivElementIt = (*proposalDerivativeIt)->begin();
          for( int propDerivElementIndex=0; propDerivElementIndex<shapeLength; ++propDerivElementIndex, ++propDerivElementIt){
            (*propDerivElementIt)/=m_BaseStd;
          }
          (**proposalDerivativeIt)[shapeLength]/=m_CentroidXStd;
          (**proposalDerivativeIt)[shapeLength+1]/=m_CentroidYStd;
          (**proposalDerivativeIt)[shapeLength+2]/=m_CentroidZStd;
          (**proposalDerivativeIt)[shapeLength+3]/=m_SizeStd;
          if (m_ShrinkageIntensity!=0)
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu(diff)+ 1/(Beta*sigma_0^2)*diff^T* d/dmu(diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot,m_EigenVectors->transpose()*(**proposalDerivativeIt))+ dot_product(differenceVector,**proposalDerivativeIt)/(m_ShrinkageIntensity))/value; 
            CalculateCutOffDerivative(*derivativeIt,value);
          }
          else//m_ShrinkageIntensity==0
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu (diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot,m_EigenVectors->transpose()*(**proposalDerivativeIt)))/value; 
            CalculateCutOffDerivative(*derivativeIt,value);
          }
          break;
        }
      default:
        {
          elxout << "Error: Bad m_ShapeModelCalculation" << std::endl;
        }

        delete (*proposalDerivativeIt);
        // nzjacs++;
      }
    }
  }
} // end CalculateDerivative()

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::CalculateCutOffValue( MeasureType & value) const{
  if (m_CutOffValue > 0.0){
    value = vcl_log(vcl_exp(m_CutOffSharpness*value)+vcl_exp(m_CutOffSharpness*m_CutOffValue))/m_CutOffSharpness;
  }
}

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::CalculateCutOffDerivative( typename DerivativeType::element_type & derivativeElement, const MeasureType & value) const{
  if (m_CutOffValue > 0.0){
    derivativeElement*=1.0/(1.0+vcl_exp(m_CutOffSharpness*(m_CutOffValue-value)));
  }
}

/**
 * ******************* PrintSelf *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet,TMovingPointSet>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
//
//   if ( this->m_ComputeSquaredDistance )
//   {
//     os << indent << "m_ComputeSquaredDistance: True"<< std::endl;
//   }
//   else
//   {
//     os << indent << "m_ComputeSquaredDistance: False"<< std::endl;
//   }
} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef __itkStatisticalShapePointPenalty_txx
