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
#ifndef _itkStackTransform_hxx
#define _itkStackTransform_hxx

#include "itkStackTransform.h"

namespace itk
{

/**
<<<<<<< HEAD
=======
 * ********************* Constructor ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
StackTransform< TScalarType, NInputDimensions, NOutputDimensions >
::StackTransform() : Superclass( OutputSpaceDimension ),
  m_NumberOfSubTransforms( 0 ),
  m_StackSpacing( 1.0 ),
  m_StackOrigin( 0.0 )
{} // end Constructor

/**
>>>>>>> e6acf3d9 (ENH: Added more functionality to the stacktransform)
 * ************************ SetParameters ***********************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::SetParameters(const ParametersType & param)
{
  // All subtransforms should be the same and should have the same number of parameters.
  // Here we check if the number of parameters is #subtransforms * #parameters per subtransform.
  if (param.GetSize() != this->GetNumberOfParameters())
  {
    itkExceptionMacro("Number of parameters does not match the number of subtransforms * the number of parameters "
                      "per subtransform.");
  }

  // Set separate subtransform parameters
  const NumberOfParametersType numSubTransformParameters = this->m_SubTransformContainer[0]->GetNumberOfParameters();
  const auto                   numberOfSubTransforms = static_cast<unsigned>(m_SubTransformContainer.size());
  for (unsigned int t = 0; t < numberOfSubTransforms; ++t)
  {
    // NTA, split the parameter by number of subparameters
    const ParametersType subparams(&(param.data_block()[t * numSubTransformParameters]), numSubTransformParameters);
    this->m_SubTransformContainer[t]->SetParametersByValue(subparams);
  }

  this->Modified();
} // end SetParameters()


/**
 * ************************ GetParameters ***********************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetParameters() const -> const ParametersType &
{
  this->m_Parameters.SetSize(this->GetNumberOfParameters());

  // Fill params with parameters of subtransforms
  unsigned int i = 0;
  for (const auto & subTransform : m_SubTransformContainer)
  {
    const auto numberOfSubTransformParameters = this->m_SubTransformContainer[0]->GetNumberOfParameters();

    const ParametersType & subparams = subTransform->GetParameters();
    for (unsigned int p = 0; p < numberOfSubTransformParameters; ++p, ++i)
    {
      this->m_Parameters[i] = subparams[p];
    }
  }

  return this->m_Parameters;
} // end GetParameters()

/**
 * ********************* TransformPoint ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::TransformPoint(
  const InputPointType & inputPoint) const -> OutputPointType
{
  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    ippr[d] = inputPoint[d];
  }

  /** Transform point using right subtransform. */
  SubTransformOutputPointType oppr;
  const unsigned int          subt =
    std::min(static_cast<unsigned int>(this->m_SubTransformContainer.size() - 1),
             static_cast<unsigned int>(
               std::max(0, vnl_math::rnd((inputPoint[ReducedInputSpaceDimension] - m_StackOrigin) / m_StackSpacing))));
  oppr = this->m_SubTransformContainer[subt]->TransformPoint(ippr);

  /** Increase dimension of input point. */
  OutputPointType opp;
  for (unsigned int d = 0; d < ReducedOutputSpaceDimension; ++d)
  {
    opp[d] = oppr[d];
  }
  opp[ReducedOutputSpaceDimension] = inputPoint[ReducedInputSpaceDimension];

  return opp;

} // end TransformPoint()


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobian(const InputPointType &       inputPoint,
                                                                              JacobianType &               jac,
                                                                              NonZeroJacobianIndicesType & nzji) const
{
  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    ippr[d] = inputPoint[d];
  }

  /** Get Jacobian from right subtransform. */
  const unsigned int subt =
    std::min(static_cast<unsigned int>(this->m_SubTransformContainer.size() - 1),
             static_cast<unsigned int>(
               std::max(0, vnl_math::rnd((inputPoint[ReducedInputSpaceDimension] - m_StackOrigin) / m_StackSpacing))));
  SubTransformJacobianType subjac;
  this->m_SubTransformContainer[subt]->GetJacobian(ippr, subjac, nzji);

  /** Fill output Jacobian. */
  jac.set_size(InputSpaceDimension, nzji.size());
  jac.Fill(0.0);
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    for (unsigned int n = 0; n < nzji.size(); ++n)
    {
      jac[d][n] = subjac[d][n];
    }
  }

  /** Update non zero Jacobian indices. */
  for (unsigned int i = 0; i < nzji.size(); ++i)
  {
    nzji[i] += subt * this->m_SubTransformContainer[0]->GetNumberOfParameters();
  }

} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
StackTransform< TScalarType, NInputDimensions, NOutputDimensions >
::GetSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Jacobian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformSpatialJacobianType sjr;
  this->m_SubTransformContainer[ subt ]->GetSpatialJacobian( ippr, sjr );

  sj.Fill( 0.0 );

  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    for( unsigned int e = 0; e < ReducedInputSpaceDimension; ++e )
    {
      sj[ d ][ e ] = sjr[ d ][ e ];
    }
  }

  sj[ ReducedInputSpaceDimension ][ ReducedInputSpaceDimension ] = 1.0;

} // end GetSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
StackTransform< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{

  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Jacobian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformTypeJacobianOfSpatialJacobianType subjacspjac;
  this->m_SubTransformContainer[ subt ]->GetJacobianOfSpatialJacobian( ippr, subjacspjac, nonZeroJacobianIndices );

  jsj.resize( this->GetNumberOfNonZeroJacobianIndices() );

  for( unsigned int i = 0; i < jsj.size(); ++i )
  {
    jsj[ i ].Fill( 0.0 );
  }

  for( unsigned int n = 0; n < nonZeroJacobianIndices.size(); ++n )
  {
    for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
    {
      for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
      {
        jsj[ n ]( d, ii ) = subjacspjac[ n ]( d, ii );

      }
    }
  }

  /** Update non zero Jacobian indices. */
  for( unsigned int i = 0; i < nonZeroJacobianIndices.size(); ++i )
  {
    nonZeroJacobianIndices[ i ] += subt * this->m_SubTransformContainer[ 0 ]->GetNumberOfParameters();
  }

}


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
StackTransform< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{

  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Jacobian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformSpatialJacobianType               subjac;
  SubTransformTypeJacobianOfSpatialJacobianType subjacspjac;
  this->m_SubTransformContainer[ subt ]->GetJacobianOfSpatialJacobian( ippr, subjac, subjacspjac, nonZeroJacobianIndices );

  jsj.resize( this->GetNumberOfNonZeroJacobianIndices() );

  for( unsigned int i = 0; i < jsj.size(); ++i )
  {
    jsj[ i ].Fill( 0.0 );
  }

  for( unsigned int n = 0; n < nonZeroJacobianIndices.size(); ++n )
  {
    for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
    {
      for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
      {
        jsj[ n ]( d, ii ) = subjacspjac[ n ]( d, ii );
      }
    }
  }

  sj.Fill( 0.0 );

  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    for( unsigned int e = 0; e < ReducedInputSpaceDimension; ++e )
    {
      sj[ d ][ e ] = subjac[ d ][ e ];
    }
  }

  sj[ ReducedInputSpaceDimension ][ ReducedInputSpaceDimension ] = 1.0;

  /** Update non zero Jacobian indices. */
  for( unsigned int i = 0; i < nonZeroJacobianIndices.size(); ++i )
  {
    nonZeroJacobianIndices[ i ] += subt * this->m_SubTransformContainer[ 0 ]->GetNumberOfParameters();
  }

}


/**
 * ********************* GetSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
StackTransform< TScalarType, NDimensions, VSplineOrder >
::GetSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh ) const
{

  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Hessian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformSpatialHessianType subhes;
  this->m_SubTransformContainer[ subt ]->GetSpatialHessian( ippr, subhes );

  /** Fill output Spatial Hessian. */
  for( unsigned int i = 0; i < sh.Size(); ++i )
  {
    sh[ i ].Fill( 0.0 );
  }

  for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
  {
    for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
    {
      for( unsigned int n = 0; n < ReducedInputSpaceDimension; ++n )
      {
        sh[ n ]( d, ii ) = subhes[ n ]( d, ii );
      }
    }
  }
}


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
StackTransform< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialHessian(
  const InputPointType & ipp,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nzji ) const
{

  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Hessian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformJacobianOfSpatialHessianType subjaches;
  this->m_SubTransformContainer[ subt ]->GetJacobianOfSpatialHessian( ippr, subjaches, nzji );

  jsh.resize( this->GetNumberOfNonZeroJacobianIndices() );

  for( unsigned int i = 0; i < jsh.size(); ++i )
  {
    for( unsigned int j = 0; j < jsh[ i ].Size(); ++j )
    {
      jsh[ i ][ j ].Fill( 0.0 );
    }
  }

  for( unsigned int n = 0; n < nzji.size(); ++n )
  {
    for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
    {
      for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
      {
        for( unsigned int k = 0; k < ReducedInputSpaceDimension; ++k )
        {
          jsh[ n ][ k ]( d, ii ) = subjaches[ n ][ k ]( d, ii );
        }
      }
    }
  }

  /** Update non zero Jacobian indices. */
  for( unsigned int i = 0; i < nzji.size(); ++i )
  {
    nzji[ i ] += subt * this->m_SubTransformContainer[ 0 ]->GetNumberOfParameters();
  }

}


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
StackTransform< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nzji ) const
{

  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
  {
    ippr[ d ] = ipp[ d ];
  }

  /** Get Hessian from right subtransform. */
  const unsigned int subt
    = vnl_math_min( this->m_NumberOfSubTransforms - 1,
    static_cast< unsigned int >( vnl_math_max( 0, vnl_math_rnd( ( ipp[ ReducedInputSpaceDimension ] - m_StackOrigin ) / m_StackSpacing ) ) ) );

  SubTransformSpatialHessianType           subhes;
  SubTransformJacobianOfSpatialHessianType subjaches;
  this->m_SubTransformContainer[ subt ]->GetJacobianOfSpatialHessian( ippr, subhes, subjaches, nzji );

  jsh.resize( this->GetNumberOfNonZeroJacobianIndices() );

  for( unsigned int i = 0; i < jsh.size(); ++i )
  {
    for( unsigned int j = 0; j < jsh[ i ].Size(); ++j )
    {
      jsh[ i ][ j ].Fill( 0.0 );
    }
  }

  for( unsigned int i = 0; i < sh.Size(); ++i )
  {
    sh[ i ].Fill( 0.0 );
  }

  for( unsigned int n = 0; n < nzji.size(); ++n )
  {
    for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
    {
      for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
      {
        for( unsigned int k = 0; k < ReducedInputSpaceDimension; ++k )
        {
          jsh[ n ][ k ]( d, ii ) = subjaches[ n ][ k ]( d, ii );
        }
      }
    }
  }

  for( unsigned int ii = 0; ii < ReducedInputSpaceDimension; ++ii )
  {
    for( unsigned int d = 0; d < ReducedInputSpaceDimension; ++d )
    {
      for( unsigned int n = 0; n < ReducedInputSpaceDimension; ++n )
      {
        sh[ n ]( d, ii ) = subhes[ n ]( d, ii );
      }
    }
  }

  /** Update non zero Jacobian indices. */
  for( unsigned int i = 0; i < nzji.size(); ++i )
  {
    nzji[ i ] += subt * this->m_SubTransformContainer[ 0 ]->GetNumberOfParameters();
  }

}


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  return this->m_SubTransformContainer[0]->GetNumberOfNonZeroJacobianIndices();

} // end GetNumberOfNonZeroJacobianIndices()


} // end namespace itk

#endif
