/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedTranslationTransform.h,v $
  Language:  C++
  Date:      $Date: 2007-07-15 16:38:25 $
  Version:   $Revision: 1.36 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkAdvancedTranslationTransform_h
#define __itkAdvancedTranslationTransform_h

#include <iostream>
#include "itkAdvancedTransform.h"
#include "itkExceptionObject.h"
#include "itkMatrix.h"


namespace itk
{

/** \brief Translation transformation of a vector space (e.g. space coordinates)
 *
 * The same functionality could be obtained by using the Affine tranform,
 * but with a large difference in performace.
 *
 * \ingroup Transforms
 */
template <
  class TScalarType=double,          // Data type for scalars (float or double)
  unsigned int NDimensions=3>        // Number of dimensions
class ITK_EXPORT AdvancedTranslationTransform :
  public AdvancedTransform< TScalarType, NDimensions, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef AdvancedTranslationTransform Self;
  typedef AdvancedTransform< TScalarType, NDimensions, NDimensions > Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** New macro for creation of through the object factory.*/
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedTranslationTransform, AdvancedTransform );

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, NDimensions);

  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Standard parameters container. */
  typedef typename Superclass::ParametersType ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::TransformCategoryType TransformCategoryType;

  /** Standard Jacobian container. */
  typedef typename Superclass::JacobianType JacobianType;

  /** Standard vector type for this class. */
  typedef Vector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputVectorType;
  typedef Vector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputVectorType;

  /** Standard covariant vector type for this class. */
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputCovariantVectorType;
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputCovariantVectorType;

  /** Standard vnl_vector type for this class. */
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputVnlVectorType;
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputVnlVectorType;

  /** Standard coordinate point type for this class. */
  typedef Point<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputPointType;
  typedef Point<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputPointType;

  /** AdvancedTransform typedefs */
  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** This method returns the value of the offset of the
   * AdvancedTranslationTransform.
   */
  const OutputVectorType & GetOffset(void) const
    { return m_Offset; }

  /** This method sets the parameters for the transform
   * value specified by the user. */
  void SetParameters(const ParametersType & parameters);

  /** Get the Transformation Parameters. */
  virtual const ParametersType& GetParameters(void) const;

  /** Set offset of an Translation Transform.
   * This method sets the offset of an AdvancedTranslationTransform to a
   * value specified by the user. */
  void SetOffset(const OutputVectorType &offset)
    { m_Offset = offset; return; }

  /** Compose with another AdvancedTranslationTransform. */
  void Compose(const Self * other, bool pre=0);

  /** Compose affine transformation with a translation.
   * This method modifies self to include a translation of the
   * origin.  The translation is precomposed with self if pre is
   * true, and postcomposed otherwise. */
  void Translate(const OutputVectorType &offset, bool pre=0);

  /** Transform by an affine transformation.
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector. */
  OutputPointType     TransformPoint(const InputPointType  &point ) const;
  OutputVectorType    TransformVector(const InputVectorType &vector) const;
  OutputVnlVectorType TransformVector(const InputVnlVectorType &vector) const;
  OutputCovariantVectorType TransformCovariantVector(
    const InputCovariantVectorType &vector) const;

  /** This method finds the point or vector that maps to a given
   * point or vector under the affine transformation defined by
   * self.  If no such point exists, an exception is thrown. */
  inline InputPointType    BackTransform(const OutputPointType  &point ) const;
  inline InputVectorType   BackTransform(const OutputVectorType &vector) const;
  inline InputVnlVectorType BackTransform(const OutputVnlVectorType &vector) const;
  inline InputCovariantVectorType BackTransform(
    const OutputCovariantVectorType &vector) const;

  /** Find inverse of an affine transformation.
   * This method creates and returns a new AdvancedTranslationTransform object
   * which is the inverse of self.  If self is not invertible,
   * false is returned.  */
  bool GetInverse(Self* inverse) const;

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType & ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType &,
    SpatialHessianType & ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;

    /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType &,
    JacobianOfSpatialHessianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Set the parameters to the IdentityTransform */
  void SetIdentity(void);

  /** Return the number of parameters that completely define the Transform  */
  virtual NumberOfParametersType GetNumberOfParameters( void ) const
  { return NDimensions; }

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear() const { return true; }

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  virtual TransformCategoryType GetTransformCategory() const
  {
    return Self::Linear;
  }

  /** Set the fixed parameters and update internal transformation.
   * The Translation Transform does not require fixed parameters,
   * therefore the implementation of this method is a null operation. */
  virtual void SetFixedParameters( const ParametersType & )
    { /* purposely blank */ };

  /** Get the Fixed Parameters. The AdvancedTranslationTransform does not
    * require Fixed parameters, therefore this method returns an
    * parameters array of size zero. */
  virtual const ParametersType& GetFixedParameters(void) const
    {
    this->m_FixedParameters.SetSize(0);
    return this->m_FixedParameters;
    };

protected:
  AdvancedTranslationTransform();
  ~AdvancedTranslationTransform();
  /** Print contents of an AdvancedTranslationTransform. */
  void PrintSelf(std::ostream &os, Indent indent) const;

private:
  AdvancedTranslationTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  OutputVectorType   m_Offset;       // Offset of the transformation

  JacobianType                  m_LocalJacobian;
  SpatialJacobianType           m_SpatialJacobian;
  SpatialHessianType            m_SpatialHessian;
  NonZeroJacobianIndicesType    m_NonZeroJacobianIndices;
  JacobianOfSpatialJacobianType m_JacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType  m_JacobianOfSpatialHessian;

}; //class AdvancedTranslationTransform


// Back transform a point
template<class TScalarType, unsigned int NDimensions>
inline
typename AdvancedTranslationTransform<TScalarType, NDimensions>::InputPointType
AdvancedTranslationTransform<TScalarType, NDimensions>::
BackTransform(const OutputPointType &point) const {
  return point - m_Offset;
}




// Back transform a vector
template<class TScalarType, unsigned int NDimensions>
inline
typename AdvancedTranslationTransform<TScalarType, NDimensions>::InputVectorType
AdvancedTranslationTransform<TScalarType, NDimensions>::
BackTransform(const OutputVectorType &vect ) const
{
  return  vect;
}




// Back transform a vnl_vector
template<class TScalarType, unsigned int NDimensions>
inline
typename AdvancedTranslationTransform<TScalarType, NDimensions>::InputVnlVectorType
AdvancedTranslationTransform<TScalarType, NDimensions>::
BackTransform(const OutputVnlVectorType &vect ) const
{
  return  vect;
}


// Back Transform a CovariantVector
template<class TScalarType, unsigned int NDimensions>
inline
typename AdvancedTranslationTransform<TScalarType, NDimensions>::InputCovariantVectorType
AdvancedTranslationTransform<TScalarType, NDimensions>::
BackTransform(const OutputCovariantVectorType &vect) const
{
  return vect;
}

}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedTranslationTransform.txx"
#endif

#endif /* __itkAdvancedTranslationTransform_h */
