/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-06-29 12:58:58 $
  Version:   $Revision: 1.64 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedTransform_h
#define __itkAdvancedTransform_h

#include "itkTransform.h"
#include "itkMatrix.h"
#include "itkFixedArray.h"

namespace itk
{

/** \class AdvancedTransform
 * \brief Transform maps points, vectors and covariant vectors from an input
 * space to an output space.
 *
 * This abstract class define the generic interface for a geometrical
 * transformation from one space to another. The class provides methods
 * for mapping points, vectors and covariant vectors from the input space
 * to the output space.
 *
 * Given that transformation are not necessarily invertible, this basic
 * class does not provide the methods for back transformation. Back transform
 * methods are implemented in derived classes where appropriate.
 *
 * \par Registration Framework Support
 * Typically a Transform class has several methods for setting its
 * parameters. For use in the registration framework, the parameters must
 * also be represented by an array of doubles to allow communication
 * with generic optimizers. The Array of transformation parameters is set using
 * the SetParameters() method.
 *
 * Another requirement of the registration framework is the computation
 * of the Jacobian of the transform T. In general, an ImageToImageMetric
 * requires the knowledge of this Jacobian in order to compute the metric
 * derivatives. The Jacobian is a matrix whose element are the partial
 * derivatives of the transformation with respect to the array of parameters
 * mu that defines the transform, evaluated at a point p: dT/dmu(p).
 *
 * If penalty terms are included in the registration, the transforms also
 * need to implement other derivatives of T. Often, penalty terms are functions
 * of the spatial derivatives of T. Therefore, e.g. the SpatialJacobian dT/dx
 * and the SpatialHessian d^2T/dx_idx_j require implementation. The
 * GetValueAndDerivative() requires the d/dmu of those terms. Therefore,
 * we additionally define GetJacobianOfSpatialJacobian() and
 * GetJacobianOfSpatialHessian().
 *
 * \ingroup Transforms
 *
 */
template< class TScalarType,
unsigned int NInputDimensions  = 3,
unsigned int NOutputDimensions = 3 >
class AdvancedTransform :
  public Transform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedTransform Self;
  typedef Transform< TScalarType,
    NInputDimensions,
    NOutputDimensions >               Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New method for creating an object using a factory. */
  //itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedTransform, Transform );

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NInputDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NOutputDimensions );

  /** Typedefs from the Superclass. */
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::ParametersValueType    ParametersValueType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::DerivativeType         DerivativeType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::InputVectorType        InputVectorType;
  typedef typename Superclass::OutputVectorType       OutputVectorType;
  typedef typename Superclass
    ::InputCovariantVectorType InputCovariantVectorType;
  typedef typename Superclass
    ::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType  InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;
  typedef typename Superclass::InputPointType      InputPointType;
  typedef typename Superclass::OutputPointType     OutputPointType;

  typedef typename Superclass::InverseTransformBaseType    InverseTransformBaseType;
  typedef typename Superclass::InverseTransformBasePointer InverseTransformBasePointer;

  /** Transform typedefs for the from Superclass. */
  typedef Transform< TScalarType,
    NInputDimensions,
    NOutputDimensions >                             TransformType;
  typedef typename TransformType::Pointer      TransformTypePointer;
  typedef typename TransformType::ConstPointer TransformTypeConstPointer;

  /** Types for the (Spatial)Jacobian/Hessian.
   * Using an itk::FixedArray instead of an std::vector gives a performance
   * gain for the SpatialHessianType.
   */
  typedef std::vector< unsigned long > NonZeroJacobianIndicesType;
  typedef Matrix< ScalarType,
    OutputSpaceDimension, InputSpaceDimension >     SpatialJacobianType;
  typedef std::vector< SpatialJacobianType > JacobianOfSpatialJacobianType;
  // \todo: think about the SpatialHessian type, should be a 3D native type
  typedef FixedArray<
    Matrix< ScalarType,
    InputSpaceDimension, InputSpaceDimension >,
    OutputSpaceDimension >                          SpatialHessianType;
  typedef std::vector< SpatialHessianType >                JacobianOfSpatialHessianType;
  typedef typename SpatialJacobianType::InternalMatrixType InternalMatrixType;

  /** Typedef for the moving image gradient type.
   * This type is defined by the B-spline interpolator as
   * typedef CovariantVector< RealType, ImageDimension >
   * As we cannot access this type we simply re-construct it to be identical.
   */
  typedef OutputCovariantVectorType                   MovingImageGradientType;
  typedef typename MovingImageGradientType::ValueType MovingImageGradientValueType;

  /** Get the number of nonzero Jacobian indices. By default all. */
  virtual NumberOfParametersType GetNumberOfNonZeroJacobianIndices( void ) const;

  /** Whether the advanced transform has nonzero matrices. */
  itkGetConstMacro( HasNonZeroSpatialHessian, bool );
  itkGetConstMacro( HasNonZeroJacobianOfSpatialHessian, bool );

  /** This returns a sparse version of the Jacobian of the transformation.
   *
   * The Jacobian is expressed as a vector of partial derivatives of the
   * transformation components with respect to the parameters \f$\mu\f$ that
   * define the transformation \f$T\f$, evaluated at a point \f$p\f$.
   *
   * \f[
      J=\left[ \begin{array}{cccc}
      \frac{\partial T_{1}}{\partial \mu_{1}}(p) &
      \frac{\partial T_{1}}{\partial \mu_{2}}(p) &
      \cdots &
      \frac{\partial T_{1}}{\partial \mu_{m}}(p) \\
      \frac{\partial T_{2}}{\partial \mu_{1}}(p) &
      \frac{\partial T_{2}}{\partial \mu_{2}}(p) &
      \cdots &
      \frac{\partial T_{2}}{\partial \mu_{m}}(p) \\
      \vdots & \vdots & \ddots & \vdots \\
      \frac{\partial T_{d}}{\partial \mu_{1}}(p) &
      \frac{\partial T_{d}}{\partial \mu_{2}}(p) &
      \cdots &
      \frac{\partial T_{d}}{\partial \mu_{m}}(p)
      \end{array}\right],
   * \f]
   * with \f$m\f$ the number of parameters, i.e. the size of \f$\mu\f$, and \f$d\f$
   * the dimension of the image.
   */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & j,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const = 0;

  /** Compute the inner product of the Jacobian with the moving image gradient.
   * The Jacobian is (partially) constructed inside this function, but not returned.
   */
  virtual void EvaluateJacobianWithImageGradientProduct(
    const InputPointType & ipp,
    const MovingImageGradientType & movingImageGradient,
    DerivativeType & imageJacobian,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Jacobian of the transformation.
   *
   * The spatial Jacobian is expressed as a vector of partial derivatives of the
   * transformation components with respect to the spatial position \f$x\f$,
   * evaluated at a point \f$p\f$.
   *
   * \f[
      sJ=\left[ \begin{array}{cccc}
      \frac{\partial T_{1}}{\partial x_{1}}(p) &
      \frac{\partial T_{1}}{\partial x_{2}}(p) &
      \cdots &
      \frac{\partial T_{1}}{\partial x_{m}}(p) \\
      \frac{\partial T_{2}}{\partial x_{1}}(p) &
      \frac{\partial T_{2}}{\partial x_{2}}(p) &
      \cdots &
      \frac{\partial T_{2}}{\partial x_{m}}(p) \\
      \vdots & \vdots & \ddots & \vdots \\
      \frac{\partial T_{d}}{\partial x_{1}}(p) &
      \frac{\partial T_{d}}{\partial x_{2}}(p) &
      \cdots &
      \frac{\partial T_{d}}{\partial x_{m}}(p)
      \end{array}\right],
   * \f]
   * with \f$m\f$ the number of parameters, i.e. the size of \f$\mu\f$, and \f$d\f$
   * the dimension of the image.
   */
  virtual void GetSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const = 0;

  /** Override some pure virtual ITK4 functions. */
  virtual void ComputeJacobianWithRespectToParameters(
    const InputPointType & itkNotUsed( p ), JacobianType & itkNotUsed( j ) ) const
  {
    itkExceptionMacro( << "This ITK4 function is currently not used in elastix." );
  }


  virtual void ComputeJacobianWithRespectToPosition(
    const InputPointType & itkNotUsed( p ), JacobianType & itkNotUsed( j ) ) const
  {
    itkExceptionMacro( << "This ITK4 function is currently not used in elastix." );
  }


  /** Compute the spatial Hessian of the transformation.
   *
   * The spatial Hessian is the vector of matrices of partial second order
   * derivatives of the transformation components with respect to the spatial
   * position \f$x\f$, evaluated at a point \f$p\f$.
   *
   * \f[
      sH=\left[ \begin{array}{cc}
      \frac{\partial^2 T_{i}}{\partial x_{1} \partial x_{1}}(p) &
      \frac{\partial^2 T_{i}}{\partial x_{1} \partial x_{2}}(p) \\
      \frac{\partial^2 T_{i}}{\partial x_{1} \partial x_{2}}(p) &
      \frac{\partial^2 T_{i}}{\partial x_{2} \partial x_{2}}(p) \\
      \end{array}\right],
   * \f]
   * with i the i-th component of the transformation.
   */
  virtual void GetSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const = 0;

  /** Compute the Jacobian of the spatial Jacobian of the transformation.
   *
   * The Jacobian of the spatial Jacobian is the derivative of the spatial
   * Jacobian to the transformation parameters \f$\mu\f$, evaluated at
   * a point \f$p\f$.
   */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const = 0;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const = 0;

  /** Compute the Jacobian of the spatial Hessian of the transformation.
   *
   * The Jacobian of the spatial Hessian is the derivative of the spatial
   * Hessian to the transformation parameters \f$\mu\f$, evaluated at
   * a point \f$p\f$.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const = 0;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const = 0;

protected:

  AdvancedTransform();
  AdvancedTransform( NumberOfParametersType numberOfParameters );
  virtual ~AdvancedTransform() {}

  bool m_HasNonZeroSpatialHessian;
  bool m_HasNonZeroJacobianOfSpatialHessian;

private:

  AdvancedTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );    // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedTransform.hxx"
#endif

#endif
