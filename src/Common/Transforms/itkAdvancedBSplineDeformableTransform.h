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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedBSplineDeformableTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-04-11 16:28:11 $
  Version:   $Revision: 1.38 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedBSplineDeformableTransform_h
#define __itkAdvancedBSplineDeformableTransform_h

#include "itkAdvancedBSplineDeformableTransformBase.h"
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkBSplineInterpolationWeightFunction2.h"
#include "itkBSplineInterpolationDerivativeWeightFunction.h"
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"

namespace itk
{

// Forward declarations for friendship
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
class MultiBSplineDeformableTransformWithNormal;

/** \class AdvancedBSplineDeformableTransform
 * \brief Deformable transform using a B-spline representation
 *
 * This class encapsulates a deformable transform of points from one
 * N-dimensional one space to another N-dimensional space.
 * The deformation field is modeled using B-splines.
 * A deformation is defined on a sparse regular grid of control points
 * \f$ \vec{\lambda}_j \f$ and is varied by defining a deformation
 * \f$ \vec{g}(\vec{\lambda}_j) \f$ of each control point.
 * The deformation \f$ D(\vec{x}) \f$ at any point \f$ \vec{x} \f$
 * is obtained by using a B-spline interpolation kernel.
 *
 * The deformation field grid is defined by a user specified GridRegion,
 * GridSpacing and GridOrigin. Each grid/control point has associated with it
 * N deformation coefficients \f$ \vec{\delta}_j \f$, representing the N
 * directional components of the deformation. Deformation outside the grid
 * plus support region for the B-spline interpolation is assumed to be zero.
 *
 * Additionally, the user can specified an addition bulk transform \f$ B \f$
 * such that the transformed point is given by:
 * \f[ \vec{y} = B(\vec{x}) + D(\vec{x}) \f]
 *
 * The parameters for this transform is N x N-D grid of spline coefficients.
 * The user specifies the parameters as one flat array: each N-D grid
 * is represented by an array in the same way an N-D image is represented
 * in the buffer; the N arrays are then concatentated together on form
 * a single array.
 *
 * For efficiency, this transform does not make a copy of the parameters.
 * It only keeps a pointer to the input parameters and assumes that the memory
 * is managed by the caller.
 *
 * The following illustrates the typical usage of this class:
 * \verbatim
 * typedef AdvancedBSplineDeformableTransform<double,2,3> TransformType;
 * TransformType::Pointer transform = TransformType::New();
 *
 * transform->SetGridRegion( region );
 * transform->SetGridSpacing( spacing );
 * transform->SetGridOrigin( origin );
 *
 * // NB: the region must be set first before setting the parameters
 *
 * TransformType::ParametersType parameters(
 *                                       transform->GetNumberOfParameters() );
 *
 * // Fill the parameters with values
 *
 * transform->SetParameters( parameters )
 *
 * outputPoint = transform->TransformPoint( inputPoint );
 *
 * \endverbatim
 *
 * An alternative way to set the B-spline coefficients is via array of
 * images. The grid region, spacing and origin information is taken
 * directly from the first image. It is assumed that the subsequent images
 * are the same buffered region. The following illustrates the API:
 * \verbatim
 *
 * TransformType::ImageConstPointer images[2];
 *
 * // Fill the images up with values
 *
 * transform->SetCoefficientImages( images );
 * outputPoint = transform->TransformPoint( inputPoint );
 *
 * \endverbatim
 *
 * Warning: use either the SetParameters() or SetCoefficientImages()
 * API. Mixing the two modes may results in unexpected results.
 *
 * The class is templated coordinate representation type (float or double),
 * the space dimension and the spline order.
 *
 * \ingroup Transforms
 */
template<
class TScalarType         = double,      // Data type for scalars
unsigned int NDimensions  = 3,           // Number of dimensions
unsigned int VSplineOrder = 3 >
// Spline order
class AdvancedBSplineDeformableTransform :
  public AdvancedBSplineDeformableTransformBase< TScalarType, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedBSplineDeformableTransform Self;
  typedef AdvancedBSplineDeformableTransformBase<
    TScalarType, NDimensions >                      Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedBSplineDeformableTransform, AdvancedBSplineDeformableTransformBase );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** The B-spline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Typedefs from Superclass. */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::ParametersValueType    ParametersValueType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::DerivativeType         DerivativeType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename Superclass::InputVectorType        InputVectorType;
  typedef typename Superclass::OutputVectorType       OutputVectorType;
  typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
  typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;
  typedef typename Superclass::MovingImageGradientType MovingImageGradientType;
  typedef typename Superclass::MovingImageGradientValueType MovingImageGradientValueType;

  /** Parameters as SpaceDimension number of images. */
  typedef typename Superclass::PixelType    PixelType;
  typedef typename Superclass::ImageType    ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;

  /** Typedefs for specifying the extend to the grid. */
  typedef typename Superclass::RegionType RegionType;

  typedef typename Superclass::IndexType      IndexType;
  typedef typename Superclass::SizeType       SizeType;
  typedef typename Superclass::SpacingType    SpacingType;
  typedef typename Superclass::DirectionType  DirectionType;
  typedef typename Superclass::OriginType     OriginType;
  typedef typename Superclass::GridOffsetType GridOffsetType;

  /** This method specifies the region over which the grid resides. */
  virtual void SetGridRegion( const RegionType & region );

  /** Transform points by a B-spline deformable transformation. */
  OutputPointType TransformPoint( const InputPointType & point ) const;

  /** Interpolation weights function type. */
  typedef BSplineInterpolationWeightFunction2< ScalarType,
    itkGetStaticConstMacro( SpaceDimension ),
    itkGetStaticConstMacro( SplineOrder ) >                 WeightsFunctionType;
  typedef typename WeightsFunctionType::Pointer             WeightsFunctionPointer;
  typedef typename WeightsFunctionType::WeightsType         WeightsType;
  typedef typename WeightsFunctionType::ContinuousIndexType ContinuousIndexType;
  typedef BSplineInterpolationDerivativeWeightFunction<
    ScalarType,
    itkGetStaticConstMacro( SpaceDimension ),
    itkGetStaticConstMacro( SplineOrder ) >                 DerivativeWeightsFunctionType;
  typedef typename DerivativeWeightsFunctionType::Pointer DerivativeWeightsFunctionPointer;
  typedef BSplineInterpolationSecondOrderDerivativeWeightFunction<
    ScalarType,
    itkGetStaticConstMacro( SpaceDimension ),
    itkGetStaticConstMacro( SplineOrder ) >                 SODerivativeWeightsFunctionType;
  typedef typename SODerivativeWeightsFunctionType::Pointer SODerivativeWeightsFunctionPointer;

  /** Parameter index array type. */
  typedef typename Superclass::ParameterIndexArrayType ParameterIndexArrayType;

  /** Transform points by a B-spline deformable transformation.
   * On return, weights contains the interpolation weights used to compute the
   * deformation and indices of the x (zeroth) dimension coefficient parameters
   * in the support region used to compute the deformation.
   * Parameter indices for the i-th dimension can be obtained by adding
   * ( i * this->GetNumberOfParametersPerDimension() ) to the indices array.
   */
  virtual void TransformPoint(
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices,
    bool & inside ) const;

  /** Get number of weights. */
  unsigned long GetNumberOfWeights( void ) const
  {
    return this->m_WeightsFunction->GetNumberOfWeights();
  }


  unsigned int GetNumberOfAffectedWeights( void ) const;

  virtual NumberOfParametersType GetNumberOfNonZeroJacobianIndices( void ) const;

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & j,
    NonZeroJacobianIndicesType & nzji ) const;

  /** Compute the inner product of the Jacobian with the moving image gradient.
   * The Jacobian is (partially) constructed inside this function, but not returned.
   */
  virtual void EvaluateJacobianWithImageGradientProduct(
    const InputPointType & ipp,
    const MovingImageGradientType & movingImageGradient,
    DerivativeType & imageJacobian,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

protected:

  /** Print contents of an AdvancedBSplineDeformableTransform. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  AdvancedBSplineDeformableTransform();
  virtual ~AdvancedBSplineDeformableTransform();

  /** Allow subclasses to access and manipulate the weights function. */
  // Why??
  itkSetObjectMacro( WeightsFunction, WeightsFunctionType );
  itkGetObjectMacro( WeightsFunction, WeightsFunctionType );

  /** Wrap flat array into images of coefficients. */
  void WrapAsImages( void );

  virtual void ComputeNonZeroJacobianIndices(
    NonZeroJacobianIndicesType & nonZeroJacobianIndices,
    const RegionType & supportRegion ) const;

  typedef typename Superclass::JacobianImageType JacobianImageType;
  typedef typename Superclass::JacobianPixelType JacobianPixelType;

  /** Pointer to function used to compute B-spline interpolation weights.
   * For each direction we create a different weights function for thread-
   * safety.
   */
  WeightsFunctionPointer                                           m_WeightsFunction;
  std::vector< DerivativeWeightsFunctionPointer >                  m_DerivativeWeightsFunctions;
  std::vector< std::vector< SODerivativeWeightsFunctionPointer > > m_SODerivativeWeightsFunctions;

private:

  AdvancedBSplineDeformableTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );                     // purposely not implemented

  friend class MultiBSplineDeformableTransformWithNormal< ScalarType,
  itkGetStaticConstMacro( SpaceDimension ),
  itkGetStaticConstMacro( SplineOrder ) >;

};

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedBSplineDeformableTransform.hxx"
#endif

#endif /* __itkAdvancedBSplineDeformableTransform_h */
