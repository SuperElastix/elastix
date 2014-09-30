/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkRecursiveBSplineTransform_h
#define __itkRecursiveBSplineTransform_h

#include "itkBSplineBaseTransform.h"
#include "itkRecursiveBSplineInterpolationWeightFunction.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkBSplineInterpolationWeightFunction2.h"
#include "itkBSplineInterpolationDerivativeWeightFunction.h"
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"

namespace itk
{
/** \class RecursiveBSplineTransform
 * \brief Deformable transform using a BSpline representation
 *
 *
 *
 * \endverbatim
 *
 * Warning: use either the SetParameters() or SetCoefficientImages()
 * API. Mixing the two modes may results in unexpected results.
 *
 * The class is templated coordinate representation type (float or double),
 * the space dimension and the spline order.
 *
 * \ingroup ITKTransform
 * \wikiexample{Registration/ImageRegistrationMethodBSpline,
 *   A global registration of two images}
 */
template <typename TScalarType = double,
          unsigned int NDimensions = 3,
          unsigned int VSplineOrder = 3>
class RecursiveBSplineTransform :
  public AdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder >
{
public:
  /** Standard class typedefs. */	
  typedef RecursiveBSplineTransform                                  Self;
    typedef AdvancedBSplineDeformableTransform<
      TScalarType, NDimensions, VSplineOrder >                       Superclass;
  typedef SmartPointer<Self>										 Pointer;
  typedef SmartPointer<const Self>									 ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineTransform, AdvancedBSplineDeformableTransform );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** The BSpline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType ScalarType;
  typedef typename Superclass::ParametersType ParametersType;
  typedef typename Superclass::ParametersValueType    ParametersValueType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::InputVectorType  InputVectorType;
  typedef typename Superclass::OutputVectorType OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType  InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;
  typedef typename Superclass::InputPointType  InputPointType;
  typedef typename Superclass::OutputPointType OutputPointType;

  /** Parameters as SpaceDimension number of images. */
  typedef typename Superclass::PixelType             PixelType;
  typedef typename Superclass::ImageType             ImageType;
  typedef typename Superclass::ImagePointer          ImagePointer;
  //typedef typename Superclass::CoefficientImageArray CoefficientImageArray;

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

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

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

  typedef itk::RecursiveBSplineInterpolationWeightFunction<TScalarType, NDimensions, VSplineOrder> RecursiveBSplineWeightFunctionType;

  //using Superclass::TransformPoint;
  virtual OutputPointType  TransformPoint( const InputPointType & point ) const;

  virtual void TransformPoint( const InputPointType & inputPoint, OutputPointType & outputPoint,
    WeightsType & weights, ParameterIndexArrayType & indices, bool & inside ) const;

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
    NonZeroJacobianIndicesType & ) const;

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

  RecursiveBSplineTransform();
  virtual ~RecursiveBSplineTransform();

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
  typename RecursiveBSplineWeightFunctionType::Pointer m_RecursiveBSplineWeightFunction;

private:

  RecursiveBSplineTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

}; // end class BSplineTransform 

//Recursive interpolation
template <unsigned int SpaceDimension, unsigned int SplineOrder, class TScalar> class interpolateFunction
{
public: 
    static inline TScalar interpolate( const TScalar * source,const long * steps,const double * weights,
		const TScalar *basePointer, Array<unsigned long>  & indices, unsigned int &c )
    {
        TScalar value = 0.0;
        for (unsigned int k = 0; k <= SplineOrder; k++)
        {
            const TScalar * a = source + steps[ k + (SpaceDimension-1)*(SplineOrder+1) ];
            value += interpolateFunction<SpaceDimension-1, SplineOrder, TScalar>::
                interpolate( a, steps, weights, basePointer, indices, c) * weights[ k + (SpaceDimension-1)*(SplineOrder+1) ];
        }
        return value;
    } 

    static inline void interpolateValueAndDerivative( TScalar derivativeAndValue[],
                                         const TScalar * source,
                                         const long * steps,
                                         const double * weights,
                                         const double * derivativeWeights)
    {
        /** derivativeAndValue length must be at least dim+1
          */
        TScalar derivativeAndValueNext[SpaceDimension+1];

        for(unsigned int n= 0; n <= SpaceDimension; ++n)
        {
            derivativeAndValue[n] = 0.0;
        }

        for (unsigned int k = 0; k <= SplineOrder; k++)
        {
            const TScalar * a = source + steps[ k + (SpaceDimension-1)*(SplineOrder+1) ];

            interpolateFunction<SpaceDimension-1, SplineOrder, TScalar>::
                    interpolateValueAndDerivative(derivativeAndValueNext, a, steps, weights, derivativeWeights);
            for(unsigned int n = 0; n < SpaceDimension; ++n)
            {
                derivativeAndValue[n] += derivativeAndValueNext[n]*weights[ k + (SpaceDimension-1)*(SplineOrder+1) ];
            }
            derivativeAndValue[SpaceDimension] += derivativeAndValueNext[0]*
                    derivativeWeights[ k + (SpaceDimension-1)*(SplineOrder+1) ];
        }
    }

};


/** End cases of the sample functions. A poitner to the coefficients is returned. */
template <unsigned int SplineOrder, class TScalar> class interpolateFunction<0, SplineOrder,TScalar>
{
public:
    static inline TScalar interpolate( const TScalar * source,
                                       const long * steps,
                                       const double * weights,
                                       const TScalar *basePointer,
                                       Array<unsigned long> & indices,
                                       unsigned int &c)
    {
		indices[c] = source-basePointer;
		++c;
        return *source;
    }

    static inline void interpolateValueAndDerivative(TScalar derivativeAndValue[],
                                                       const TScalar * source,
                                                       const long * steps,
                                                       const double * weights,
                                                       const double * derivativeWeights)
    {
        derivativeAndValue[0] = *source;
    }

};//end template 


}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineTransform.hxx"
#endif

#endif /* __itkRecursiveBSplineTransform_h */
