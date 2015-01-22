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
#ifndef __itkGPUNearestNeighborInterpolateImageFunction_h
#define __itkGPUNearestNeighborInterpolateImageFunction_h

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkVersion.h"
#include "itkImage.h"

#include "itkGPUInterpolateImageFunction.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUNearestNeighborInterpolateImageFunction */
itkGPUKernelClassMacro( GPUNearestNeighborInterpolateImageFunctionKernel );

/** \class GPUNearestNeighborInterpolateImageFunction
 * \brief GPU version of NearestNeighborInterpolateImageFunction.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template< typename TInputImage, typename TCoordRep = float >
class ITK_EXPORT GPUNearestNeighborInterpolateImageFunction :
  public         GPUInterpolateImageFunction< TInputImage, TCoordRep,
  NearestNeighborInterpolateImageFunction< TInputImage, TCoordRep > >
{
public:

  /** Standard class typedefs. */
  typedef GPUNearestNeighborInterpolateImageFunction                           Self;
  typedef NearestNeighborInterpolateImageFunction< TInputImage, TCoordRep >    CPUSuperclass;
  typedef GPUInterpolateImageFunction< TInputImage, TCoordRep, CPUSuperclass > GPUSuperclass;
  typedef SmartPointer< Self >                                                 Pointer;
  typedef SmartPointer< const Self >                                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUNearestNeighborInterpolateImageFunction, GPUSuperclass );

protected:

  GPUNearestNeighborInterpolateImageFunction();
  ~GPUNearestNeighborInterpolateImageFunction() {}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const ITK_OVERRIDE;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  virtual bool GetSourceCode( std::string & source ) const ITK_OVERRIDE;

private:

  GPUNearestNeighborInterpolateImageFunction( const Self & ); // purposely not implemented
  void operator=( const Self & );                             // purposely not implemented

  std::vector< std::string > m_Sources;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUNearestNeighborInterpolateImageFunction.hxx"
#endif

#endif /* __itkGPUNearestNeighborInterpolateImageFunction_h */
