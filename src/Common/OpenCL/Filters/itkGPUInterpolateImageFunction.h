/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUInterpolateImageFunction_h
#define __itkGPUInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkGPUInterpolatorBase.h"

namespace itk
{
/** \class GPUInterpolateImageFunction
 */
template< class TInputImage, class TCoordRep = float, class TParentImageFilter =
  InterpolateImageFunction< TInputImage, TCoordRep > >
class ITK_EXPORT GPUInterpolateImageFunction
  : public TParentImageFilter, public GPUInterpolatorBase
{
public:
  /** Standard class typedefs. */
  typedef GPUInterpolateImageFunction Self;
  typedef TParentImageFilter          Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUInterpolateImageFunction, TParentImageFilter);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
    TInputImage::ImageDimension);

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** CoordRepType typedef support. */
  typedef typename Superclass::CoordRepType CoordRepType;

protected:
  GPUInterpolateImageFunction();
  ~GPUInterpolateImageFunction() {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  virtual GPUDataManager::Pointer GetParametersDataManager() const;

private:
  GPUInterpolateImageFunction(const Self &); //purposely not implemented
  void operator=(const Self &);              //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUInterpolateImageFunction.hxx"
#endif

#endif
