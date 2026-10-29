// SDK 1.40.0's ambient sandbox file re-exports a private package subpath.
// Point the runtime module names at Adobe's public, exported type entrypoints.
// No API is invented or widened here.
declare module 'express-document-sdk' {
  export * from '@adobe/ccweb-add-on-sdk-types/express-document-sdk';
}
declare module 'add-on-sdk-document-sandbox' {
  import { AddOnDocumentSandboxSdkTypes } from '@adobe/ccweb-add-on-sdk-types';
  export default AddOnDocumentSandboxSdkTypes.default;
  export * from '@adobe/ccweb-add-on-sdk-types/add-on-sdk-document-sandbox';
}
