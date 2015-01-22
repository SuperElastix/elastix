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
#ifndef __itkGPUAdvancedCombinationTransform_h
#define __itkGPUAdvancedCombinationTransform_h

#include "itkAdvancedCombinationTransform.h"
#include "itkGPUCompositeTransformBase.h"

namespace itk
{
/** \class GPUAdvancedCombinationTransform
 * \brief GPU version of AdvancedCombinationTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template< typename TScalarType = float, unsigned int NDimensions = 3,
typename TParentTransform      = AdvancedCombinationTransform< TScalarType, NDimensions > >
class GPUAdvancedCombinationTransform :
  public TParentTransform, public GPUCompositeTransformBase< TScalarType, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef GPUAdvancedCombinationTransform                       Self;
  typedef TParentTransform                                      CPUSuperclass;
  typedef GPUCompositeTransformBase< TScalarType, NDimensions > GPUSuperclass;
  typedef SmartPointer< Self >                                  Pointer;
  typedef SmartPointer< const Self >                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedCombinationTransform, TParentTransform );

  /** Sub transform types. */
  typedef typename GPUSuperclass::TransformType             GPUTransformType;
  typedef typename GPUSuperclass::TransformTypePointer      TransformTypePointer;
  typedef typename GPUSuperclass::TransformTypeConstPointer TransformTypeConstPointer;

  /** Get number of transforms in composite transform. */
  virtual SizeValueType GetNumberOfTransforms( void ) const
  { return CPUSuperclass::GetNumberOfTransforms(); }

  /** Get the Nth transform. */
  virtual const TransformTypePointer GetNthTransform( SizeValueType n ) const
  { return CPUSuperclass::GetNthTransform( n ); }

protected:

  GPUAdvancedCombinationTransform() {}
  virtual ~GPUAdvancedCombinationTransform() {}
  void PrintSelf( std::ostream & s, Indent indent ) const ITK_OVERRIDE
  { CPUSuperclass::PrintSelf( s, indent ); }

private:

  GPUAdvancedCombinationTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                // purposely not implemented

};

} // end namespace itk

#endif /* __itkGPUAdvancedCombinationTransform_h */
