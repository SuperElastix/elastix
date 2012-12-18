/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAffineTransform_h
#define __itkGPUAffineTransform_h

#include "itkAffineTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAffineTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3,
          class TParentImageFilter = AffineTransform< TScalarType, NDimensions > >
class GPUAffineTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, NDimensions, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUAffineTransform Self;
  typedef TParentImageFilter CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType,
                                        NDimensions,
                                        NDimensions > GPUSuperclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAffineTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUAffineTransform() {}
  virtual ~GPUAffineTransform() {}

private:
  GPUAffineTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );   // purposely not implemented
};

/** \class GPUAffineTransformFactory
* \brief Object Factory implementation for GPUAffineTransform
*/
class GPUAffineTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAffineTransformFactory  Self;
  typedef ObjectFactoryBase          Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAffineTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAffineTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAffineTransformFactory::Pointer factory =
      GPUAffineTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAffineTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

#define OverrideAffineTransformTypeMacro( st, dm )                   \
  {                                                                  \
    this->RegisterOverride(                                          \
      typeid( AffineTransform< st, dm > ).name(),                    \
      typeid( GPUAffineTransform< st, dm > ).name(),                 \
      "GPU AffineTransform Override",                                \
      true,                                                          \
      CreateObjectFunction< GPUAffineTransform< st, dm > >::New() ); \
  }

  GPUAffineTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAffineTransformTypeMacro( float, 1 );
      //OverrideAffineTransformTypeMacro(double, 1);

      OverrideAffineTransformTypeMacro( float, 2 );
      //OverrideAffineTransformTypeMacro(double, 2);

      OverrideAffineTransformTypeMacro( float, 3 );
      //OverrideAffineTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUAffineTransform_h */
